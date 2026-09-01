from __future__ import annotations

from pathlib import Path

import pandas as pd

from .... import cache_path
from ... import client
from .ourairports import DownloadedAirportsProvider


class Fr24AirportsProvider(DownloadedAirportsProvider):
    def __init__(self) -> None:
        super().__init__(
            source="flightradar24",
            name="flightradar24",
            priority=90,
            confidence=0.8,
        )

    @property
    def cache_file(self) -> Path:
        return cache_path / "airports_fr24.parquet"

    def fetch_data(self) -> pd.DataFrame:
        # Flightradar24 sits behind Cloudflare, which blocks the bare
        # "Mozilla/5.0" user-agent with an HTTP 403 challenge page: the body
        # is HTML, so response.json() raises JSONDecodeError and the airports
        # source appears "no longer available".  Browser-like headers make the
        # endpoint return the JSON payload again.
        response = client.get(
            "https://www.flightradar24.com/_json/airports.php",
            headers={
                "user-agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "accept": "application/json, text/plain, */*",
                "accept-language": "en-US,en;q=0.9",
                "referer": "https://www.flightradar24.com/",
            },
        )
        data = pd.DataFrame.from_records(response.json()["rows"])
        data = data.assign(name=data.name.str.strip())
        data = data.rename(
            columns={
                "lat": "latitude",
                "lon": "longitude",
                "alt": "altitude",
            }
        )
        data["altitude"] = pd.to_numeric(
            data["altitude"], errors="coerce"
        ).fillna(0)
        data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
        data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
        return data
