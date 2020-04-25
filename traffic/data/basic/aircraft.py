import io
import json
import logging
import re
import zipfile
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from tqdm.autonotebook import tqdm

json_path = Path(__file__).parent / "patterns.json"

registration_patterns = list(
    dict(
        (k, int(v[2:], 16) if k in ["start", "end"] else v)
        for (k, v) in elt.items()
    )
    for elt in json.loads(json_path.read_text())["registers"]
)


def country(reg: Dict[str, str]) -> Dict[str, str]:

    # First, search the country based on the registered address intervals
    icao24 = int(reg["icao24"], 16)
    candidate = next(
        (
            elt
            for elt in registration_patterns
            if "start" in elt.keys() and elt["start"] <= icao24 <= elt["end"]
        ),
        None,
    )

    # If not found or suspicious (Unassigned), look at the tail number pattern
    if (
        candidate is None or candidate["country"].startswith("Unassigned")
    ) and "registration" in reg.keys():
        candidate = next(
            (
                elt
                for elt in registration_patterns
                if "pattern" in elt
                and re.match(elt["pattern"], reg["registration"])
            ),
            None,
        )

    # Still nothing? Give up...
    if candidate is None:
        return {"country": "Unknown", "flag": "🏳", "tooltip": "Unknown"}

    # It could be possible to be more specific with categories
    # Also some tail numbers are attributed to different countries within
    #   the same ICAO address range

    if "registration" in reg.keys() and "categories" in candidate.keys():
        precise = next(
            (
                elt
                for elt in candidate["categories"]
                if "pattern" in elt
                and re.match(elt["pattern"], reg["registration"])
            ),
            None,
        )
        if precise is not None:
            candidate = {**candidate, **precise}

    return candidate


class Aircraft(object):
    """

    `@junzis <https://github.com/junzis/>`_'s `database
    <https://junzisun.com/adb/download>`_ is available by default in the
    library as:

    >>> from traffic.data import aircraft

    Basic requests can be made by the bracket notation:

    >>> aircraft["F-GFKY"]
          icao24 registration typecode             model    operator
    3032  391558       F-GFKY     A320   Airbus A320-211  Air France
    >>> aircraft["391558"]
          icao24 registration typecode             model    operator
    3032  391558       F-GFKY     A320   Airbus A320-211  Air France

    A more comprehensive database can be manually downloaded or upgraded (the
    operation can take up to five minutes with a slow Internet connection):

    >>> aircraft.download_opensky()



    """

    cache_dir: Path
    basic_columns = ["icao24", "registration", "typecode", "model", "operator"]

    def __init__(self):
        self._junzis: Optional[pd.DataFrame] = None
        self._opensky: Optional[pd.DataFrame] = None
        self._merged: Optional[pd.DataFrame] = None

    def download_junzis(self) -> None:  # coverage: ignore
        """Downloads the latest version of the aircraft database by @junzis.

        url: https://junzisun.com/adb/download
        """
        from .. import session

        f = session.get("https://junzisun.com/adb/download")
        with zipfile.ZipFile(io.BytesIO(f.content)) as zfile:
            with zfile.open("aircraft_db.csv", "r") as dbfile:
                self._junzis = (
                    pd.read_csv(dbfile, dtype=str)
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
    def junzis_db(self) -> pd.DataFrame:  # coverage: ignore
        if self._junzis is None:
            if not (self.cache_dir / "junzis_db.pkl").exists():
                self.download_junzis()
            else:
                logging.info("Loading @junzis aircraft database")
                self._junzis = pd.read_pickle(self.cache_dir / "junzis_db.pkl")

        assert self._junzis is not None
        return self._junzis.fillna("")

    def download_opensky(self):  # coverage: ignore
        """Downloads the latest version of the OpenSky aircraft database.

        url: https://opensky-network.org/aircraft-database
        """
        from .. import session

        logging.warning("Downloading OpenSky aircraft database")
        file_url = (
            "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
        )
        f = session.get(file_url, stream=True)
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

        # coverage: ignore
        # ignore that part below, a correct DB depends on this part anyway

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

    def get_unique(self, name: str) -> Optional[Dict[str, str]]:
        df = self[name]

        if df.shape[0] == 0:
            return None

        return {**dict(df.iloc[0]), **country(dict(df.iloc[0]))}

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
        """Requests an aircraft by owner or operator (fuzzy match).

        The query string may match the owner or operator (full name, ICAO
        code or IATA code)

        >>> aircraft.operator("Speedbird").head()
                icao24 registration typecode serialnumber           model
        7041    400409       G-BNLJ     B744        24052  Boeing 747-436
        7042    40040a       G-BNLK     B744        24053  Boeing 747-436
        7043    40040d       G-BNLN     B744        24056  Boeing 747-436
        7044    40040e       G-BNLO     B744        24057  Boeing 747-436
        7045    40040f       G-BNLP     B744        24058  Boeing 747-436
                       operator operatoricao            owner
        7041    British Airways          BAW  British Airways
        7042    British Airways          BAW  British Airways
        7043    British Airways          BAW  British Airways
        7044    British Airways          BAW  British Airways
        7045    British Airways          BAW  British Airways

        """
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
        """Computes stats of owned or operated aircraft (fuzzy match).

        >>> aircraft.stats("HOP")
                              model  icao24
        typecode
        AT45             ATR 42-500      13
        AT75             ATR 72-500       9
        AT76             ATR 72-600       5
        CRJ1       Canadair CRJ-100       5
        CRJ7       Canadair CRJ-700      13
        CRJX      Canadair CRJ-1000      14
        E145      Embraer ERJ-145MP      17
        E170        Embraer ERJ-170      16
        E190        Embraer ERJ-190      10

        """
        return (
            self.operator(name)
            .drop_duplicates("icao24")
            .groupby("typecode")
            .agg(dict(model="max", icao24="count"))
        )

    def model(self, name: str) -> pd.DataFrame:
        """Requests an aircraft by model or typecode (fuzzy match).

        >>> aircraft.model("RJ85").head()
             icao24 registration typecode serialnumber       model
        39   0081fb       ZS-ASW     RJ85        E2313   Avro RJ85
        40   0081fc       ZS-ASX     RJ85        E2314   Avro RJ85
        41   0081fd       ZS-ASY     RJ85        E2316   Avro RJ85
        42   0081fe       ZS-ASZ     RJ85        E2318   Avro RJ85
        240  00b174       ZS-SSH     RJ85        E2285   Avro RJ85
            operator operatoricao                  owner
        39   Airlink               South African Airlink
        40   Airlink               South African Airlink
        41   Airlink               South African Airlink
        42   Airlink               South African Airlink
        240  Airlink               South African Airlink
        """
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
        """Requests an aircraft by registration (fuzzy match).

        >>> aircraft.registration("F-ZB").sample(5)
                icao24 registration typecode serialnumber
        5617    3b7b77       F-ZBAB     F406     406-0025
        454188  3b7b6c       F-ZBFX     CL4T         2007
        94647   3b7ba4       F-ZBPG     EC45         9013
        245275  3b7b21       F-ZBGP     B350       FL-802
        5626    3b7ba5       F-ZBPF     EC45         9012
                                     model operator            owner
        5617           Reims F406 Vigilant            French Customs
        454188                    CL-415 T           Securite Civile
        94647       MBB-BK 117 C-2 (EC145)           Securite Civile
        245275               King Air B350            French Customs
        5626    Eurocopter-Kawasaki EC-145           Securite Civile

        """
        return self._fmt(self.data.query("registration.str.contains(@name)"))

    def query(self, **kwargs) -> pd.DataFrame:
        """Combines several requests.

        The keyword arguments correspond to the name of other methods.

        >>> aircraft.query(registration="^F-ZB", model="EC45").head()
           icao24 registration typecode serialnumber                   model
        2  3b7b4a       F-ZBQK     EC45         9372  MBB-BK 117 C-2 (EC145)
        3  3b7b4b       F-ZBQJ     EC45         9323  MBB-BK 117 C-2 (EC145)
        4  3b7b50       F-ZBQI     EC45         9240  MBB-BK 117 C-2 (EC145)
        5  3b7b51       F-ZBQH     EC45         9232  MBB-BK 117 C-2 (EC145)
        6  3b7b52       F-ZBQG     EC45         9217  MBB-BK 117 C-2 (EC145)
                          operator operatoricao            owner
        2          Securite Civile               Securite Civile
        3          Securite Civile               Securite Civile
        4          Securite Civile               Securite Civile
        5          Securite Civile               Securite Civile
        6          Securite Civile               Securite Civile
        """
        subs = (getattr(self, key)(value) for key, value in kwargs.items())
        res = reduce(lambda a, b: a.merge(b, how="inner"), subs)
        return res
