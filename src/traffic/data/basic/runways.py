from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

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


@dataclass
class Threshold(PointMixin):
    latitude: float
    longitude: float
    bearing: float
    name: str
    # necessary for being a PointLike (but not used)
    altitude: float = float("nan")

    def __repr__(self) -> str:
        return f"Runway {self.name}: {self.latlon}"


class Runway(HBoxMixin, ShapelyMixin, DataFrameMixin):
    def __init__(self, tuple_runway: tuple[Threshold, Threshold]):
        self.tuple_runway = tuple_runway

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            self.tuple_runway,
            columns=["latitude", "longitude", "bearing", "name"],
        )

    def geojson(self) -> Dict[str, Any]:
        return {
            "type": "LineString",
            "coordinates": tuple(
                (thrs.longitude, thrs.latitude) for thrs in self.tuple_runway
            ),
        }

    @property
    def shape(self) -> base.BaseGeometry:
        return shape(self.geojson())

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

            for thr in self.tuple_runway:
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
            return super().geoencode().mark_geoshape(**params)  # type: ignore
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
                rwy_labels.transform_filter(alt.datum.name == name).mark_text(
                    angle=bearing, **params
                )
                for (name, bearing) in zip(self.data.name, self.data.bearing)
            ]

            return alt.layer(*rwy_layers)  # type: ignore
        elif kwargs.get("mode", None) is None:
            return alt.layer(  # type: ignore
                self.geoencode(mode="geometry", **kwargs),
                self.geoencode(mode="labels", **kwargs),
            )

        raise ValueError("mode must be 'geometry' or 'labels'")


class RunwaysAirport(HBoxMixin, ShapelyMixin, DataFrameMixin):
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        runways: List[Tuple[Threshold, Threshold]] = list(),
    ) -> None:
        self._data: Optional[pd.DataFrame] = data
        self._runways = runways

    def __getitem__(self, runway_id: str) -> Runway:
        elt = next(
            (t for t in self._runways if any(x.name == runway_id for x in t)),
            None,
        )
        if elt:
            return Runway(elt)
        raise ValueError(f"Runway {runway_id} not found")

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [vars(thr) for thr in self.list],
            columns=["latitude", "longitude", "bearing", "name"],
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
            return super().geoencode().mark_geoshape(**params)  # type: ignore
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
                rwy_labels.transform_filter(alt.datum.name == name).mark_text(
                    angle=bearing, **params
                )
                for (name, bearing) in zip(self.data.name, self.data.bearing)
            ]

            return alt.layer(*rwy_layers)  # type: ignore

        raise ValueError("mode must be 'geometry' or 'labels'")


class Runways(object):
    cache_path: Optional[Path] = None

    def __init__(self) -> None:
        self._runways: Optional[pd.DataFrame] = None
        assert self.cache_path is not None
        self._cache = self.cache_path / "runways_ourairports.parquet"

    @property
    def runways(self) -> pd.DataFrame:
        if self._runways is not None:
            return self._runways

        if not self._cache.exists():
            self.download_runways()

        last_modification = self._cache.lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        if cache_expiration is not None and delta > cache_expiration:
            try:
                self.download_runways()
            except httpx.TransportError:
                pass

        self._runways = pd.read_parquet(self._cache)
        return self._runways

    def __getitem__(self, name: Union["Airport", str]) -> RunwaysAirport:
        from .. import airports

        airport_ = airports[name] if isinstance(name, str) else name
        icao = airport_.icao

        _df = self.runways.query(f"airport_ident =='{icao}'")
        cur: List[Tuple[Threshold, Threshold]] = list()
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

        if cur is None:
            raise AttributeError(f"Runway information not found for {name}")
        return RunwaysAirport(runways=cur)

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
        df.to_parquet(self._cache)
