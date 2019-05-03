import io
import logging
import zipfile
from functools import reduce
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import requests
from tqdm.autonotebook import tqdm


class Aircraft(object):

    cache_dir: Path
    basic_columns = ["icao24", "registration", "typecode", "model", "operator"]

    def __init__(self):
        self._junzis: Optional[pd.DataFrame] = None
        self._opensky: Optional[pd.DataFrame] = None
        self._merged: Optional[pd.DataFrame] = None

    def download_junzis(self) -> None:
        """Downloads the latest version of the aircraft database by @junzis.

        url: https://junzisun.com/adb/download
        """
        f = requests.get("https://junzisun.com/adb/download")
        with zipfile.ZipFile(io.BytesIO(f.content)) as zfile:
            with zfile.open("aircraft_db.csv", "r") as dbfile:
                self._junzis = (
                    pd.read_csv(dbfile)
                    .assign(
                        regid=lambda df: df.regid.str.upper(),
                        mdl=lambda df: df.mdl.str.upper(),
                    )
                    .rename(
                        columns={
                            "icao": "icao24",
                            "regid": "registration",
                            "mdl": "typecode",
                            "type": "model",
                        }
                    )
                )
                self._junzis.to_pickle(self.cache_dir / "junzis_db.pkl")

        if (self.cache_dir / "merged_db.pkl").exists():
            (self.cache_dir / "merged_db.pkl").unlink()

    @property
    def junzis_db(self) -> pd.DataFrame:
        if self._junzis is None:
            if not (self.cache_dir / "junzis_db.pkl").exists():
                self.download_junzis()
            else:
                logging.info("Loading @junzis aircraft database")
                self._junzis = pd.read_pickle(self.cache_dir / "junzis_db.pkl")

        assert self._junzis is not None
        return self._junzis.fillna("")

    def download_opensky(self):
        """Downloads the latest version of the OpenSky aircraft database.

        url: https://opensky-network.org/aircraft-database
        """
        logging.warn("Downloading OpenSky aircraft database")
        file_url = (
            "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
        )
        f = requests.get(file_url, stream=True)
        total = int(f.headers["Content-Length"])
        buffer = io.BytesIO()
        for chunk in tqdm(
            f.iter_content(1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="download",
        ):
            buffer.write(chunk)

        buffer.seek(0)
        self._opensky = pd.read_csv(
            buffer,
            dtype={"icao24": str, "operator": str},
            skiprows=[1],
            engine="c",
            keep_default_na=False,
        )
        self._opensky.to_pickle(self.cache_dir / "opensky_db.pkl")

        if (self.cache_dir / "merged_db.pkl").exists():
            (self.cache_dir / "merged_db.pkl").unlink()

    @property
    def opensky_db(self) -> Optional[pd.DataFrame]:
        if self._opensky is None:
            if not (self.cache_dir / "opensky_db.pkl").exists():
                return None
            else:
                logging.info("Loading OpenSky aircraft database")
                self._opensky = pd.read_pickle(
                    self.cache_dir / "opensky_db.pkl"
                )
        return self._opensky

    @property
    def data(self) -> pd.DataFrame:
        if self._merged is not None:
            return self._merged

        if (self.cache_dir / "merged_db.pkl").exists():
            logging.info("Loading merged aircraft database")
            self._merged = pd.read_pickle(self.cache_dir / "merged_db.pkl")
            return self._merged

        if not (self.cache_dir / "opensky_db.pkl").exists():
            return self.junzis_db

        self._merged = self.junzis_db.merge(
            self.opensky_db,
            on=["icao24", "registration", "typecode"],
            how="outer",
            suffixes=("_jz", "_os"),
        ).fillna("")

        self._merged.to_pickle(self.cache_dir / "merged_db.pkl")

        return self._merged

    def __getitem__(self, name: Union[str, List[str]]) -> pd.DataFrame:
        """Requests an aircraft by icao24 or registration (exact match)."""

        if isinstance(name, str):
            df = self.data.query(
                "icao24 == @name.lower() or registration == @name.upper()"
            )
        else:
            df = self.data.query("icao24 in @name or registration in @name")
        return self._fmt(df)

    def _fmt(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("icao24")
        if self.opensky_db is None:
            return df[self.basic_columns]
        else:

            df = df.assign(model=df.model_jz, operator=df.operator_jz)
            df.loc[
                df.model_jz.str.len() < df.model_os.str.len(), "model"
            ] = df.model_os

            df.loc[
                df.operator_jz.str.len() < df.operator_os.str.len(), "operator"
            ] = df.operator_os

            return df[
                [
                    "icao24",
                    "registration",
                    "typecode",
                    "serialnumber",
                    "model",
                    "operator",
                    "operatoricao",
                    "owner",
                ]
            ]

    def operator(self, name: str) -> pd.DataFrame:
        """Requests an aircraft by owner or operator (fuzzy match)."""
        if self.opensky_db is None:
            return self._fmt(self.data.query("operator.str.contains(@name)"))
        else:
            return self._fmt(
                self.data.query(
                    "operator_jz.str.contains(@name) or "
                    "operator_os.str.contains(@name) or "
                    "@name.upper() == operatoricao or "
                    "@name.upper() == operatoriata or "
                    "@name.upper() == operatorcallsign or "
                    "owner.str.contains(@name)"
                )
            )

    def stats(self, name: str) -> pd.DataFrame:
        """Computes stats of owned or operated aircraft (fuzzy match)."""
        return (
            self.operator(name)
            .drop_duplicates("icao24")
            .groupby("typecode")
            .agg(dict(model="max", icao24="count"))
        )

    def model(self, name: str) -> pd.DataFrame:
        """Requests an aircraft by model or typecode (fuzzy match)."""
        if self.opensky_db is None:
            return self._fmt(
                self.data.query(
                    "model.str.contains(@name) or "
                    "typecode.str.contains(@name.upper())"
                )
            )
        else:
            return self._fmt(
                self.data.query(
                    "model_jz.str.contains(@name) or "
                    "model_os.str.contains(@name) or "
                    "typecode.str.contains(@name.upper())"
                )
            )

    def registration(self, name: str) -> pd.DataFrame:
        """Requests an aircraft by registration (fuzzy match)."""
        return self._fmt(self.data.query("registration.str.contains(@name)"))

    def query(self, **kwargs) -> pd.DataFrame:
        """Combines several requests."""
        subs = (getattr(self, key)(value) for key, value in kwargs.items())
        res = reduce(lambda a, b: a.merge(b, how="inner"), subs)
        return res
