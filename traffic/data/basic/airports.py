# flake8: noqa

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import requests
from tqdm.autonotebook import tqdm

import pandas as pd

from ... import cache_expiration
from ...core.mixins import GeoDBMixin
from ...core.structure import Airport

__all__ = ["Airport", "Airports"]


class Airports(GeoDBMixin):
    """
    An airport is accessible via its ICAO or IATA code. In case of doubt,
    use the search method.

    The representation of an airport is based on its geographical footprint.
    Contours are fetched from OpenStreetMap (you need an Internet connection the
    first time you call it) and put in cache.

    A database of major world airports is available as:

    >>> from traffic.data import airports

    Any airport can be accessed by the bracket notation:

    >>> airports["EHAM"]
    EHAM/AMS: Amsterdam Airport Schiphol
    >>> airports["EHAM"].latlon
    (52.308609, 4.763889)
    >>> airports["EHAM"].iata
    AMS
    >>> airports["EHAM"].name
    Amsterdam Airport Schiphol

    Runways thresholds are also associated to most airports:

    >>> airports['EHAM'].runways
      latitude   longitude   bearing   name
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      52.3       4.783       41.36     04
      52.31      4.803       221.4     22
      52.29      4.734       57.93     06
      52.3       4.778       238       24
      52.32      4.746       86.65     09
      52.32      4.797       266.7     27
      52.33      4.74        183       18C
      52.3       4.737       2.997     36C
      52.32      4.78        183       18L
      52.29      4.777       3.002     36R
    ... (2 more entries)

    """

    cache_dir: Path
    expiration_days: Optional[int]

    src_dict = dict(
        fr24=("airports_fr24.pkl", "download_fr24"),
        open=("airports_ourairports.pkl", "download_airports"),
    )

    columns_options = dict(
        name=dict(),
        country=dict(justify="right"),
        icao=dict(style="blue bold"),
        iata=dict(),
        latitude=dict(justify="left", max_width=10),
        longitude=dict(justify="left", max_width=10),
    )

    def __init__(self, data: None | pd.DataFrame = None) -> None:
        self._data: Optional[pd.DataFrame] = data
        self._src = "open"

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

        f = session.get("https://ourairports.com/data/countries.csv")
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

        >>> airports.query('type == "large_airport"').search('Tokyo')
          name                                 country   icao   iata   latitude   longitude
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Narita International Airport           Japan   RJAA   NRT    35.76      140.4
          Tokyo Haneda International Airport     Japan   RJTT   HND    35.55      139.8

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
            )
        else:
            return self.__class__(
                self.data.query(
                    "iata == @name.upper() or "
                    "icao.str.contains(@name.upper()) or "
                    "country.str.upper().str.contains(@name.upper()) or "
                    "name.str.upper().str.contains(@name.upper())"
                ),
            )
