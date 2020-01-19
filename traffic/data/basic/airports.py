# flake8: noqa

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd

import altair as alt

from ...core.mixins import GeoDBMixin, PointMixin, ShapelyMixin
from ...core.structure import Airport


class Airports(GeoDBMixin):
    """
    An airport is accessible via its ICAO or IATA code. In case of doubt,
    use the search method.

    The representation of an airport is based on its geographical footprint.
    It subclasses namedtuple so all fields are accessible by the dot
    operator. It can also be displayed on Matplotlib maps. Contours are
    fetched from OpenStreetMap (you need an Internet connection the first
    time you call it) and put in cache.

    A database of major world airports is available as:

    >>> from traffic.data import airports

    Any airport can be accessed by the bracket notation:

    >>> airports["EHAM"]
    EHAM/AMS: Amsterdam Schiphol
    >>> airports["EHAM"].latlon
    (52.308609, 4.763889)
    >>> airports["EHAM"].iata
    AMS

    Runways thresholds are also associated to most airports:

    >>> airports['LFPG'].runways.data
        latitude  longitude     bearing name
    0  48.995664   2.552155   85.379257  08L
    1  48.998757   2.610603  265.423366  26R
    2  49.024736   2.524890   85.399341  09L
    3  49.026678   2.561694  265.427128  27R
    4  48.992929   2.565816   85.399430  08R
    5  48.994863   2.602438  265.427067  26L
    6  49.020645   2.513055   85.391641  09R
    7  49.023665   2.570303  265.434861  27L
    """

    cache_dir: Path

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self._data: Optional[pd.DataFrame] = data

    def download_fr24(self) -> None:  # coverage: ignore
        from .. import session

        c = session.get(
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

    def search(self, name: str) -> "Airports":
        """
        Selects the subset of airports matching the given IATA or ICAO code,
        containing the country name or the full name of the airport.

        >>> airports.search('Tokyo')
            altitude country iata  icao   latitude   longitude                                name
        3820       21   Japan  HND  RJTT  35.552250  139.779602  Tokyo Haneda International Airport
        3821      135   Japan  NRT  RJAA  35.764721  140.386307  Tokyo Narita International Airport

        """
        return self.__class__(
            self.data.query(
                "iata == @name.upper() or "
                "icao.str.contains(@name.upper()) or "
                "country.str.upper().str.contains(@name.upper()) or "
                "name.str.upper().str.contains(@name.upper())"
            )
        )
