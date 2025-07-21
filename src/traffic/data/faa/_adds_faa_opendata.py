import json
import logging
from typing import Any, Dict

import httpx

import pandas as pd

from ... import cache_expiration, cache_path
from .. import client

_log = logging.getLogger(__name__)


class ADDS_FAA_OpenData:
    id_: str
    filename: str
    website = "https://adds-faa.opendata.arcgis.com/datasets/{}"
    json_url = "https://opendata.arcgis.com/datasets/{}.geojson"

    def __init__(self) -> None:
        self.cache_file = cache_path / self.filename
        self.website = self.website.format(self.id_)
        self.json_url = self.json_url.format(self.id_)

    def download_data(self) -> None:
        _log.warning(
            f"Downloading data from {self.website}. Please check terms of use."
        )
        c = client.get(self.json_url)
        c.raise_for_status()
        json_contents = c.json()
        with self.cache_file.open("w") as fh:
            json.dump(json_contents, fh)

    def json_contents(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            last_modification = (self.cache_file).lstat().st_mtime
            delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)

            if cache_expiration is not None and delta > cache_expiration:
                try:
                    self.download_data()
                except httpx.TransportError:
                    pass
        else:
            self.download_data()

        with self.cache_file.open("r") as fh:
            json_contents = json.load(fh)

        return json_contents  # type: ignore
