from __future__ import annotations

import io
from pathlib import Path

import httpx

import pandas as pd

from .... import cache_expiration, cache_path
from ....core import tqdm
from ... import client
from ..models import EntityKind, ResolutionCandidate, row_payload


class DownloadedAirportsProvider:
    kinds = ("airport",)
    source: str
    name: str
    priority: int
    confidence: float

    def __init__(
        self,
        *,
        source: str,
        name: str,
        priority: int,
        confidence: float,
    ) -> None:
        self.source = source
        self.name = name
        self.priority = priority
        self.confidence = confidence
        self._data: None | pd.DataFrame = None

    @property
    def cache_file(self) -> Path:
        raise NotImplementedError

    def fetch_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _refresh_cache(self) -> None:
        data = self.fetch_data()
        data.to_parquet(self.cache_file)

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        if not self.cache_file.exists():
            self._refresh_cache()

        last_modification = self.cache_file.lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        if cache_expiration is not None and delta > cache_expiration:
            try:
                self._refresh_cache()
            except (httpx.HTTPError, ValueError):
                # A stale cache is available: tolerate transport failures and
                # non-JSON error pages (e.g. a Cloudflare 403 challenge) by
                # falling back to the cached snapshot instead of propagating.
                pass

        self._data = pd.read_parquet(self.cache_file)
        return self._data

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind is not None and kind != "airport":
            return []

        upper = code.upper()
        matches = self.data.query("icao == @upper or iata == @upper")

        return [
            ResolutionCandidate(
                code=(row.get("icao") or row.get("iata") or upper),
                kind="airport",
                source=self.source,
                confidence=self.confidence,
                payload=row_payload(row),
            )
            for _, row in matches.iterrows()
        ]


class OurAirportsProvider(DownloadedAirportsProvider):
    def __init__(self) -> None:
        super().__init__(
            source="ourairports",
            name="ourairports",
            priority=50,
            confidence=0.7,
        )

    @property
    def cache_file(self) -> Path:
        return cache_path / "airports_ourairports.parquet"

    def fetch_data(self) -> pd.DataFrame:
        f = client.get("https://ourairports.com/data/airports.csv")
        total = int(f.headers.get("Content-Length", "0"))
        buffer = io.BytesIO()
        for chunk in tqdm(
            f.iter_bytes(chunk_size=1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="airports @ourairports.com",
        ):
            buffer.write(chunk)

        buffer.seek(0)
        airports_data = pd.read_csv(buffer)

        f = client.get("https://ourairports.com/data/countries.csv")
        buffer = io.BytesIO(f.content)
        buffer.seek(0)
        countries = pd.read_csv(buffer)

        return airports_data.rename(
            columns={
                "latitude_deg": "latitude",
                "longitude_deg": "longitude",
                "elevation_ft": "altitude",
                "iata_code": "iata",
                "ident": "icao",
            }
        ).merge(
            countries[["code", "name"]].rename(
                columns={"code": "iso_country", "name": "country"}
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
