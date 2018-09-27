import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ...core import Traffic


class Aircraft(object):

    cache: Optional[Path] = None

    def __init__(self):
        if self.cache is not None and self.cache.exists():
            self.aircraft = pd.read_pickle(self.cache)
        else:
            f = requests.get("https://junzisun.com/adb/download")
            with zipfile.ZipFile(io.BytesIO(f.content)) as zfile:
                with zfile.open("aircraft_db.csv", "r") as dbfile:
                    self.aircraft = pd.read_csv(dbfile)
                    self.aircraft.regid = self.aircraft.regid.str.upper()
                    self.aircraft.mdl = self.aircraft.mdl.str.upper()
                    self.aircraft.to_pickle(self.cache)

    def __getitem__(self, name: str) -> pd.DataFrame:
        table = self.aircraft[
            (self.aircraft.icao == name.lower())
            | (self.aircraft.regid == name.upper())
        ]
        return table

    def operator(self, name: str) -> pd.DataFrame:
        table = self.aircraft[
            self.aircraft.operator.astype(str).str.contains(name)
        ]
        return table

    def stats(self, name: str) -> pd.DataFrame:
        return self.operator(name).groupby(["mdl", "type"])[["icao"]].count()

    def merge(self, t: Traffic) -> Traffic:
        return Traffic(
            t.data.merge(
                self.aircraft.rename(columns={"icao": "icao24"}),
                on="icao24",
                how="left",
            )
        )
