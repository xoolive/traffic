import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from zipfile import ZipFile

import altair as alt
import pandas as pd
import requests
from shapely.geometry import base, shape
from shapely.ops import linemerge

from ...core.geodesy import bearing
from ...core.mixins import ShapelyMixin

__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "ProfHoekstra/bluesky/master/data/navdata"


class Threshold(NamedTuple):
    latitude: float
    longitude: float
    bearing: float
    name: str


RunwaysType = Dict[str, List[Tuple[Threshold, Threshold]]]


class RunwayAirport(ShapelyMixin):
    def __init__(self, runways: List[Tuple[Threshold, Threshold]]):
        self._runways = runways

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            self.list, columns=["latitude", "longitude", "bearing", "name"]
        )

    @property
    def list(self) -> List[Threshold]:
        return sum((list(runway) for runway in self._runways), [])

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

    def geoencode(
        self, mode: str = "geometry"
    ) -> Optional[alt.Chart]:  # coverage: ignore

        if mode == "geometry":
            return (
                super().geoencode().mark_geoshape(strokeWidth=2, stroke="black")
            )

        elif mode == "labels":
            rwy_labels = alt.Chart(self.data).encode(
                longitude="longitude:Q", latitude="latitude:Q", text="name:N"
            )
            rwy_layers = [
                rwy_labels.transform_filter(alt.datum.name == name).mark_text(
                    angle=bearing, baseline="middle", dy=10
                )
                for (name, bearing) in zip(self.data.name, self.data.bearing)
            ]

            return alt.layer(*rwy_layers)

        else:
            return None


class Runways(object):

    cache_dir: Optional[Path] = None

    def __init__(self) -> None:
        self._runways: Optional[RunwaysType] = None
        assert self.cache_dir is not None
        self._cache = self.cache_dir / "runways_bluesky.pkl"

    @property
    def runways(self) -> RunwaysType:
        if self._runways is not None:
            return self._runways

        if self._cache.exists():
            with self._cache.open("rb") as fh:
                self._runways = pickle.load(fh)
        else:
            self.download_bluesky()
            assert self._runways is not None
            with self._cache.open("wb") as fh:
                pickle.dump(self._runways, fh)

        return self._runways

    def __getitem__(self, airport) -> Optional[RunwayAirport]:
        if isinstance(airport, str):
            from .. import airports

            airport = airports[airport]
        if airport is None:
            return None
        return RunwayAirport(self.runways[airport.icao])

    def download_bluesky(self) -> None:  # coverage: ignore
        self._runways: RunwaysType = dict()

        c = requests.get(base_url + "/apt.zip")

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
                    self.runways[elems[4]] = cur

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
