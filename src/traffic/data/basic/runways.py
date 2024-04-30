from __future__ import annotations

import pickle
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from zipfile import ZipFile

import httpx
from pitot.geodesy import bearing, destination

import numpy as np
import pandas as pd
from shapely.geometry import base, shape
from shapely.ops import linemerge

from ... import cache_expiration
from ...core import tqdm
from ...core.mixins import DataFrameMixin, HBoxMixin, PointMixin, ShapelyMixin
from .. import client

if TYPE_CHECKING:
    import altair as alt
    from cartopy.mpl.geoaxes import GeoAxesSubplot

    from ...core.structure import Airport

__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "ProfHoekstra/bluesky/master/data/navdata"


class ThresholdTuple(NamedTuple):
    latitude: float
    longitude: float
    bearing: float
    name: str


class Threshold(ThresholdTuple, PointMixin):
    def __repr__(self) -> str:
        return f"Runway {self.name}: {self.latlon}"


RunwaysType = Dict[str, List[Tuple[Threshold, Threshold]]]


class RunwayAirport(HBoxMixin, ShapelyMixin, DataFrameMixin):
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        runways: List[Tuple[Threshold, Threshold]] = list(),
    ) -> None:
        self._data: Optional[pd.DataFrame] = data
        self._runways = runways

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            self.list, columns=["latitude", "longitude", "bearing", "name"]
        )

    @property
    def list(self) -> List[Threshold]:
        return list(thr for runway in self._runways for thr in runway)

    def geojson(self) -> List[Dict[str, Any]]:
        return [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": tuple(
                        (thrs.longitude, thrs.latitude) for thrs in runway
                    ),
                },
                "properties": "/".join(thrs.name for thrs in runway),
                "type": "Feature",
            }
            for runway in self._runways
        ]

    @property
    def shape(self) -> base.BaseGeometry:
        return linemerge(shape(x["geometry"]) for x in self.geojson())

    def plot(
        self,
        ax: "GeoAxesSubplot",
        *args: Any,
        runways: bool = True,
        labels: bool = False,
        shift: int = 300,
        text_kw: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:  # coverage: ignore
        from cartopy.crs import PlateCarree

        if runways is True:
            params = {
                "edgecolor": "#0e1111",
                "crs": PlateCarree(),
                "linewidth": 3,
                **kwargs,
            }
            ax.add_geometries([self.shape], **params)

        if labels is True:
            if text_kw is None:
                text_kw = dict()

            text_kw = {
                **dict(
                    transform=PlateCarree(),
                    fontsize=18,
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation_mode="anchor",
                ),
                **text_kw,
            }

            for thr in self.list:
                # Placement of labels
                lat, lon, _ = destination(
                    thr.latitude, thr.longitude, thr.bearing + 180, shift
                )

                # Compute the rotation of labels
                lat2, lon2, _ = destination(lat, lon, thr.bearing + 180, 1000)
                x1, y1 = ax.projection.transform_point(
                    thr.longitude, thr.latitude, PlateCarree()
                )
                x2, y2 = ax.projection.transform_point(
                    lon2, lat2, PlateCarree()
                )
                rotation = 90 + np.degrees(np.arctan2(y2 - y1, x2 - x1))

                ax.text(lon, lat, thr.name, rotation=rotation, **text_kw)

    def geoencode(self, **kwargs: Any) -> "alt.Chart":  # coverage: ignore
        import altair as alt

        if kwargs.get("mode", None) == "geometry":
            params = {**{"strokeWidth": 4, "stroke": "black"}, **kwargs}
            del params["mode"]
            return super().geoencode().mark_geoshape(**params)
        elif kwargs.get("mode", None) == "labels":
            params = {
                **{"baseline": "middle", "dy": 20, "fontSize": 18},
                **kwargs,
            }
            del params["mode"]
            rwy_labels = alt.Chart(self.data).encode(
                longitude="longitude:Q", latitude="latitude:Q", text="name:N"
            )
            rwy_layers = [
                rwy_labels.transform_filter(
                    alt.datum.name == name  # type: ignore
                ).mark_text(angle=bearing, **params)
                for (name, bearing) in zip(self.data.name, self.data.bearing)
            ]

            return alt.layer(*rwy_layers)  # type: ignore

        raise ValueError("mode must be 'geometry' or 'labels'")


class Runways(object):
    cache_dir: Optional[Path] = None

    def __init__(self) -> None:
        self._runways: Optional[RunwaysType] = None
        assert self.cache_dir is not None
        self._cache = self.cache_dir / "runways_ourairports.pkl"

    @property
    def runways(self) -> RunwaysType:
        if self._runways is not None:
            return self._runways

        if not self._cache.exists():
            self.download_runways()

        last_modification = self._cache.lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        if delta > cache_expiration:
            try:
                self.download_runways()
            except httpx.TransportError:
                pass

        with self._cache.open("rb") as fh:
            self._runways = pickle.load(fh)
            return self._runways

    def __getitem__(
        self, airport: Union["Airport", str]
    ) -> Optional[RunwayAirport]:
        from .. import airports

        airport_: Optional["Airport"] = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if airport_ is None:
            return None
        elt = self.runways.get(airport_.icao, None)
        if elt is None:
            return None
        return RunwayAirport(runways=elt)

    def download_runways(self) -> None:  # coverage: ignore
        self._runways = dict()

        f = client.get("https://ourairports.com/data/runways.csv")
        total = int(f.headers["Content-Length"])
        buffer = BytesIO()
        for chunk in tqdm(
            f.iter_bytes(chunk_size=1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="runways @ourairports.com",
        ):
            buffer.write(chunk)

        buffer.seek(0)
        df = pd.read_csv(buffer)

        for name, _df in df.groupby("airport_ident"):
            cur: List[Tuple[Threshold, Threshold]] = list()
            self._runways[name] = cur

            for _, line in _df.iterrows():
                lat0 = line.le_latitude_deg
                lon0 = line.le_longitude_deg
                name0 = line.le_ident
                lat1 = line.he_latitude_deg
                lon1 = line.he_longitude_deg
                name1 = line.he_ident

                if lat0 != lat0 or lat1 != lat1:
                    # some faulty records here...
                    continue

                brng0 = bearing(lat0, lon0, lat1, lon1)
                brng1 = bearing(lat1, lon1, lat0, lon0)
                brng0 = brng0 if brng0 > 0 else 360 + brng0
                brng1 = brng1 if brng1 > 0 else 360 + brng1

                thr0 = Threshold(lat0, lon0, brng0, name0)
                thr1 = Threshold(lat1, lon1, brng1, name1)
                cur.append((thr0, thr1))

        with self._cache.open("wb") as fh:
            pickle.dump(self._runways, fh)

    def download_bluesky(self) -> None:  # coverage: ignore
        self._runways = dict()
        c = client.get(base_url + "/apt.zip")

        with ZipFile(BytesIO(c.content)).open("apt.dat", "r") as fh:
            for line in fh.readlines():
                elems = (
                    line.decode(encoding="ascii", errors="ignore")
                    .strip()
                    .split()
                )
                if len(elems) == 0:
                    continue

                # 1: AIRPORT
                if elems[0] == "1":
                    # Add airport to runway threshold database
                    cur: List[Tuple[Threshold, Threshold]] = list()
                    self._runways[elems[4]] = cur

                if elems[0] == "100":
                    # Only asphalt and concrete runways
                    if int(elems[2]) > 2:
                        continue

                    lat0 = float(elems[9])
                    lon0 = float(elems[10])
                    # offset0 = float(elems[11])

                    lat1 = float(elems[18])
                    lon1 = float(elems[19])
                    # offset1 = float(elems[20])

                    # threshold information:
                    #       ICAO code airport,
                    #       Runway identifier,
                    #       latitude, longitude, bearing
                    # vertices: gives vertices of the box around the threshold

                    # opposite runways are on the same line.
                    #       RWY1: 8-11, RWY2: 17-20
                    # Hence, there are two thresholds per line
                    # thr0: First lat0 and lon0, then lat1 and lat1, offset=[11]
                    # thr1: First lat1 and lat1, then lat0 and lon0, offset=[20]

                    brng0 = bearing(lat0, lon0, lat1, lon1)
                    brng1 = bearing(lat1, lon1, lat0, lon0)
                    brng0 = brng0 if brng0 > 0 else 360 + brng0
                    brng1 = brng1 if brng1 > 0 else 360 + brng1

                    thr0 = Threshold(lat0, lon0, brng0, elems[8])
                    thr1 = Threshold(lat1, lon1, brng1, elems[17])
                    cur.append((thr0, thr1))

        with self._cache.open("wb") as fh:
            pickle.dump(self._runways, fh)
