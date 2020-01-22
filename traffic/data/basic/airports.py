# flake8: noqa

import io
import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import requests

import altair as alt
from tqdm.autonotebook import tqdm

from ... import cache_expiration
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
    expiration_days: Optional[int]

    src_dict = dict(
        fr24=("airports_fr24.pkl", "download_fr24"),
        open=("airports_ourairports.pkl", "download_airports"),
    )

    def __init__(
        self, data: Optional[pd.DataFrame] = None, src: str = "open"
    ) -> None:
        self._data: Optional[pd.DataFrame] = data
        self._src = src

    def download_airports(self) -> None:  # coverage: ignore
        from .. import session

        f = session.get(
            "https://ourairports.com/data/airports.csv", stream=True
        )
        total = int(f.headers["Content-Length"])
        buffer = io.BytesIO()
        for chunk in tqdm(
            f.iter_content(1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="airports @ourairports.com",
        ):
            buffer.write(chunk)

        buffer.seek(0)
        df = pd.read_csv(buffer)

        f = session.get("https://ourairports.com/data/countries.csv",)
        buffer = io.BytesIO(f.content)
        buffer.seek(0)
        countries = pd.read_csv(buffer)

        self._data = df.rename(
            columns={
                "latitude_deg": "latitude",
                "longitude_deg": "longitude",
                "elevation_ft": "altitude",
                "iata_code": "iata",
                "ident": "icao",
            }
        ).merge(
            countries[["code", "name"]].rename(
                columns=dict(code="iso_country", name="country")
            )
        )[
            [
                "name",
                "iata",
                "icao",
                "latitude",
                "longitude",
                "country",
                "altitude",
                "type",
                "municipality",
            ]
        ]

        self._data.to_pickle(self.cache_dir / "airports_ourairports.pkl")

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

        cache_file, method_name = self.src_dict[self._src]

        if not (self.cache_dir / cache_file).exists():
            getattr(self, method_name)()

        last_modification = (self.cache_dir / cache_file).lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        if delta > cache_expiration:
            try:
                getattr(self, method_name)()
            except requests.ConnectionError:
                pass

        self._data = pd.read_pickle(self.cache_dir / cache_file)

        return self._data

    def __getitem__(self, name: str) -> Optional[Airport]:
        x = self.data.query("iata == @name.upper() or icao == @name.upper()")
        if x.shape[0] == 0:
            return None
        p = x.iloc[0]
        return Airport(
            p.altitude,
            p.country,
            p.iata,
            p.icao,
            p.latitude,
            p.longitude,
            p["name"],
        )

    def search(self, name: str) -> "Airports":
        """
        Selects the subset of airports matching the given IATA or ICAO code,
        containing the country name or the full name of the airport.

        >>> airports.search('Tokyo')
            altitude country iata  icao   latitude   longitude                                name
        3820       21   Japan  HND  RJTT  35.552250  139.779602  Tokyo Haneda International Airport
        3821      135   Japan  NRT  RJAA  35.764721  140.386307  Tokyo Narita International Airport

        """
        if "municipality" in self.data.columns:
            return self.__class__(
                self.data.query(
                    "iata == @name.upper() or "
                    "icao.str.contains(@name.upper()) or "
                    "country.str.upper().str.contains(@name.upper()) or "
                    "municipality.str.upper().str.contains(@name.upper()) or "
                    "name.str.upper().str.contains(@name.upper())"
                ),
                src=self._src,
            )
        else:
            return self.__class__(
                self.data.query(
                    "iata == @name.upper() or "
                    "icao.str.contains(@name.upper()) or "
                    "country.str.upper().str.contains(@name.upper()) or "
                    "name.str.upper().str.contains(@name.upper())"
                ),
                src=self._src,
            )
