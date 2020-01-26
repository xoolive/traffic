import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from ... import cache_expiration
from .. import cache_dir

__all__ = list(p.stem[1:] for p in Path(__file__).parent.glob("_[a-z]*py"))


class ADDS_FAA_OpenData:

    id_: str
    filename: str
    website = "https://adds-faa.opendata.arcgis.com/datasets/{}"
    json_url = "https://opendata.arcgis.com/datasets/{}.geojson"

    def __init__(self) -> None:
        self.cache_file = cache_dir / self.filename
        self.website = self.website.format(self.id_)
        self.json_url = self.json_url.format(self.id_)

    def download_data(self) -> None:
        from .. import session

        logging.warning(
            f"Downloading data from {self.website}. Please check terms of use."
        )
        c = session.get(self.json_url)
        c.raise_for_status()
        json_contents = c.json()
        with self.cache_file.open("w") as fh:
            json.dump(json_contents, fh)

    def json_contents(self) -> Dict[str, Any]:
        if self.cache_file.exists():

            last_modification = (self.cache_file).lstat().st_mtime
            delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)

            if delta > cache_expiration:
                try:
                    self.download_data()
                except requests.ConnectionError:
                    pass
        else:
            self.download_data()

        with self.cache_file.open("r") as fh:
            json_contents = json.load(fh)

        return json_contents


def __getattr__(name: str):

    if name in __all__:
        mod = importlib.import_module("._" + name, package="traffic.data.faa")
        return getattr(mod, name.title())().get_data()

    raise AttributeError()
