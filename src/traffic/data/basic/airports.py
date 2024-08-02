# ruff: noqa: E501
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, ClassVar

import httpx

import pandas as pd

from ... import cache_expiration
from ...core import tqdm
from ...core.mixins import GeoDBMixin
from ...core.structure import Airport
from .. import client

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

    Airports information can be accessed with attributes:

    >>> airports["EHAM"].latlon  # doctest: +NUMBER
    (52.3086, 4.7639)
    >>> airports["EHAM"].iata
    'AMS'
    >>> airports["EHAM"].name
    'Amsterdam Airport Schiphol'

    """

    cache_dir: Path
    expiration_days: None | int

    src_dict: ClassVar[dict[str, tuple[str, str]]] = dict(
        fr24=("airports_fr24.parquet", "download_fr24"),
        open=("airports_ourairports.parquet", "download_airports"),
    )

    columns_options: ClassVar[dict[str, dict[str, Any]]] = dict(  # type: ignore
        name=dict(),
        country=dict(justify="right"),
        icao=dict(style="blue bold"),
        iata=dict(),
        latitude=dict(justify="left", max_width=10),
        longitude=dict(justify="left", max_width=10),
    )

    def __init__(self, data: None | pd.DataFrame = None) -> None:
        self._data: None | pd.DataFrame = data
        self._src = "open"

    def download_airports(self) -> None:  # coverage: ignore
        """
        Download an up to date version of the airports database from
        `ourairports.com <https://ourairports.com/>`_
        """

        f = client.get("https://ourairports.com/data/airports.csv")
        total = int(f.headers["Content-Length"])
        buffer = io.BytesIO()
        for chunk in tqdm(
            f.iter_bytes(chunk_size=1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="airports @ourairports.com",
        ):
            buffer.write(chunk)

        buffer.seek(0)
        df = pd.read_csv(buffer)

        f = client.get("https://ourairports.com/data/countries.csv")
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

        self._data.to_parquet(self.cache_dir / "airports_ourairports.parquet")

    def download_fr24(self) -> None:  # coverage: ignore
        c = client.get(
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
        self._data.to_parquet(self.cache_dir / "airports_fr24.parquet")

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
            except httpx.TransportError:
                pass

        self._data = pd.read_parquet(self.cache_dir / cache_file)

        return self._data

    def __getitem__(self, name: str) -> Airport:
        """
        Any airport can be accessed by the bracket notation.

        :param name: the IATA or ICAO code of the airport

        >>> from traffic.data import airports
        >>> airports["EHAM"]
        Airport(icao='EHAM', iata='AMS', name='Amsterdam Airport Schiphol', country='Netherlands', latitude=52.308601, longitude=4.76389, altitude=-11)
        """
        if isinstance(name, int):
            p = self.data.iloc[name]
        else:
            x = self.data.query(
                "iata == @name.upper() or icao == @name.upper()"
            )
            if x.shape[0] == 0:
                raise ValueError(f"Unknown airport {name} in current database")
            p = x.iloc[0]
        return Airport(
            int(p.altitude),
            p.country,
            p.iata,
            p.icao,
            float(p.latitude),
            float(p.longitude),
            p["name"],
        )

    def search(self, name: str) -> "Airports":
        """
        :param name: refers to the IATA or ICAO code, or part of the country
            name, city name of full name of the airport.


        >>> from traffic.data import airports
        >>> airports.query('type == "large_airport"').search('Tokyo')  # doctest: +SKIP
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
