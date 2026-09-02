# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, overload

import httpx

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


    Airports information can be accessed with attributes:

    >>> airports["EHAM"].latlon  # doctest: +NUMBER
    (52.3086, 4.7639)
    >>> airports["EHAM"].iata
    'AMS'
    >>> airports["EHAM"].name
    'Amsterdam Airport Schiphol'


    """

    cache_path: Path
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
        self._resolver_source: None | str = None

    def source(self, source: str) -> "Airports":
        clone = self.__class__(self._data)
        clone._src = self._src
        clone._extent = self._extent
        clone._resolver_source = source
        return clone

    @property
    def available(self) -> bool:
        return True

    def download_airports(self) -> None:  # coverage: ignore
        """
        Download an up to date version of the airports database from
        `ourairports.com <https://ourairports.com/>`_
        """
        from ..resolver import OurAirportsProvider

        self._data = OurAirportsProvider().fetch_data()
        self._data.to_parquet(self.cache_path / "airports_ourairports.parquet")

    def download_fr24(self) -> None:  # coverage: ignore
        from ..resolver import Fr24AirportsProvider

        self._data = Fr24AirportsProvider().fetch_data()
        self._data.to_parquet(self.cache_path / "airports_fr24.parquet")

    @property
    def data(self) -> pd.DataFrame:
        if self._resolver_source is not None:
            from .. import resolver

            frame = resolver.data(source=self._resolver_source, kind="airport")
            if frame.empty:
                raise RuntimeError(
                    f"No airport data found for source {self._resolver_source}"
                )
            return frame

        if self._data is not None:
            return self._data

        cache_file, method_name = self.src_dict[self._src]

        if not (self.cache_path / cache_file).exists():
            getattr(self, method_name)()

        last_modification = (self.cache_path / cache_file).lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        if cache_expiration is not None and delta > cache_expiration:
            try:
                getattr(self, method_name)()
            except httpx.TransportError:
                pass

        self._data = pd.read_parquet(self.cache_path / cache_file)

        return self._data

    @overload
    def __getitem__(self, key: str) -> Airport: ...
    @overload
    def __getitem__(self, key: Any) -> Any: ...

    def __getitem__(self, key: Any) -> Any:
        """
        Any airport can be accessed by the bracket notation.

        :param name: the IATA or ICAO code of the airport

        >>> from traffic.data import airports
        >>> airports["EHAM"]
        Airport(icao='EHAM', iata='AMS', name='Amsterdam Airport Schiphol', country='Netherlands', latitude=52.308601, longitude=4.76389, altitude=-11)

        """
        if isinstance(key, int):
            p = self.data.iloc[key]
        elif not isinstance(key, str):
            return super().__getitem__(key)
        else:
            return self.get(key)

        return Airport(
            int(p.altitude),
            p.country,
            p.iata,
            p.icao,
            float(p.latitude),
            float(p.longitude),
            p["name"],
        )

    def get(
        self,
        name: str,
        source: None | str = None,
        **kwargs: Any,
    ) -> Airport:
        selected_source = (
            source if source is not None else self._resolver_source
        )

        if selected_source is None and self._data is not None:
            upper = name.upper()
            matches = self._data[
                (self._data["icao"] == upper) | (self._data["iata"] == upper)
            ]
            if matches.empty:
                raise ValueError(f"Unknown airport {name} in current database")
            payload = matches.iloc[0]
            return Airport(
                int(payload.get("altitude", 0) or 0),
                str(payload.get("country") or ""),
                str(payload.get("iata") or ""),
                str(payload.get("icao") or ""),
                float(payload.get("latitude") or 0.0),
                float(payload.get("longitude") or 0.0),
                str(payload.get("name") or ""),
            )

        from .. import resolver

        result = resolver.resolve(
            airport=name, source=selected_source, **kwargs
        )
        if result.selected is None:
            if source is None:
                raise ValueError(f"Unknown airport {name} in current database")
            raise ValueError(f"Unknown airport {name} in source {source}")

        payload = result.selected.payload
        return Airport(
            int(payload.get("altitude", 0) or 0),
            str(payload.get("country") or ""),
            str(payload.get("iata") or ""),
            str(payload.get("icao") or ""),
            float(payload.get("latitude") or 0.0),
            float(payload.get("longitude") or 0.0),
            str(payload.get("name") or ""),
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
