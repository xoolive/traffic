from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import requests
from cartopy.crs import PlateCarree
from shapely.geometry import mapping

from ...core.mixins import PointMixin, ShapelyMixin


class AirportNamedTuple(NamedTuple):

    altitude: float
    country: str
    iata: str
    icao: str
    latitude: float
    longitude: float
    name: str


class Airport(AirportNamedTuple, PointMixin, ShapelyMixin):
    def __repr__(self):
        short_name = (
            self.name.replace("International", "")
            .replace("Airport", "")
            .strip()
        )
        return f"{self.icao}/{self.iata}: {short_name}"

    def _repr_html_(self):
        title = f"<b>{self.name.strip()}</b> ({self.country}) "
        title += f"<code>{self.icao}/{self.iata}</code>"
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    @lru_cache()
    def osm_request(self):
        from cartotools.osm import request, tags

        return request(
            (
                self.longitude - 0.06,
                self.latitude - 0.06,
                self.longitude + 0.06,
                self.latitude + 0.06,
            ),
            **tags.airport,
        )

    @lru_cache()
    def geojson(self):
        return [
            {
                "geometry": mapping(shape),
                "properties": info["tags"],
                "type": "Feature",
            }
            for info, shape in self.osm_request().ways.values()
        ]

    @property
    def shape(self):
        return self.osm_request().shape

    @property
    def runways(self):
        from .. import runways

        return runways[self.icao]

    def plot(self, ax, **kwargs):
        params = {
            "edgecolor": "silver",
            "facecolor": "None",
            "crs": PlateCarree(),
            **kwargs,
        }
        ax.add_geometries(list(self.osm_request()), **params)


class Airports(object):

    cache_dir: Path

    def __init__(self) -> None:
        self._data: Optional[pd.DataFrame] = None

    def download_fr24(self) -> None:
        c = requests.get(
            "https://www.flightradar24.com/_json/airports.php",
            headers={"user-agent": "Mozilla/5.0"},
        )

        self._data = (
            pd.DataFrame.from_records(c.json()["rows"])
            .assign(name=lambda df: df.name.str.strip())
            .rename(
                columns={
                    "lat": "latitude",
                    "lon": "longitude",
                    "alt": "altitude",
                }
            )
        )
        self._data.to_pickle(self.cache_dir / "airports_fr24.pkl")

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        if not (self.cache_dir / "airports_fr24.pkl").exists():
            self.download_fr24()

        self._data = pd.read_pickle(self.cache_dir / "airports_fr24.pkl")

        return self._data

    def __getitem__(self, name: str) -> Optional[Airport]:
        x = self.data.query("iata == @name.upper() or icao == @name.upper()")
        if x.shape[0] == 0:
            return None
        return Airport(**dict(x.iloc[0]))

    def search(self, name: str) -> pd.DataFrame:
        return self.data.query(
            "iata == @name.upper() or "
            "icao.str.contains(@name.upper()) or "
            "country.str.upper().str.contains(@name.upper()) or "
            "name.str.upper().str.contains(@name.upper())"
        )
