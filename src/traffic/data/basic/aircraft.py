# ruff: noqa: E501
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from functools import reduce
from pathlib import Path
from typing import Any, ClassVar, Dict, TypeVar

import rich.repr

import pandas as pd

from ...core import tqdm
from ...core.mixins import DataFrameMixin, FormatMixin
from .. import client

_log = logging.getLogger(__name__)

json_path = Path(__file__).parent / "patterns.json"

registration_patterns = list(
    dict(
        (k, int(v[2:], 16) if k in ["start", "end"] else v)
        for (k, v) in elt.items()
    )
    for elt in json.loads(json_path.read_text())["registers"]
)


def pia_candidate(df: pd.DataFrame) -> pd.DataFrame:
    """Returns subsets of dataframes which could be part of the PIA program.

    PIA stands for Private ICAO Address.
    Check https://doi.org/10.2514/1.I010938 for more details

    - registration should be between N41000 (0xA4D691) and N42 (0xA4F946)
    - callsign should start starts with DCM or FFL
    - registration number in the CAR is reserved by SBS PROGRAM OFFICE.

    """
    return (
        df.assign(_tmp=lambda df: df.icao24.apply(int, base=16))
        .query(
            "0xA4D691 <= _tmp < 0xA4F946 and "
            '(callsign.str.startswith("DCM") or callsign.str.startswith("FFL"))'
        )
        .drop(columns=["_tmp"])
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


@rich.repr.auto()
class Tail(Dict[str, str], FormatMixin):
    def __getattr__(self, name: str) -> None | str:
        if name in self.keys():
            return self[name]
        return None

    def __rich_repr__(self) -> rich.repr.Result:
        yield "icao24", self["icao24"]
        yield "registration", self["registration"]
        yield "typecode", self["typecode"]
        yield "flag", self["flag"]
        if "category" in self.keys():
            yield "category", self["category"]


T = TypeVar("T", bound="Aircraft")


class Aircraft(DataFrameMixin):
    """By default, the OpenSky aircraft database is downloaded from
    https://opensky-network.org/aircraft-database

    >>> from traffic.data import aircraft

    The database can be manually downloaded or upgraded (the operation can take
    up to five minutes with a slow Internet connection), with
    :meth:`~traffic.data.basic.aircraft.Aircraft.download_opensky`

    Basic requests can be made by the bracket notation, they return **a subset
    of the database, even if the result is unique**

    >>> aircraft["F-GFKY"]  # doctest: +SKIP
      icao24   registration   typecode   model             operator     owner
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      391558   F-GFKY         A320       Airbus A320 211   Air France   Air France

    >>> aircraft["391558"]  # doctest: +SKIP
      icao24   registration   typecode   model             operator   owner
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      391558   F-GFKY         A320       Airbus A320 211   Air France   Air France

    Only :meth:`~traffic.data.basic.aircraft.Aircraft.get` returns a
    :class:`~traffic.data.basic.aircraft.Tail` object.

    >>> aircraft.get("F-GFKY")
    Tail(icao24='391558', registration='F-GFKY', typecode='A320', flag='🇫🇷')

    .. tip::

        Different custom databases may also be used as a replacement if you
        provide a path in the configuration file.

    | You may set the path to any file that pandas can open (.csv, .pickle, etc.).
    | Required columns for the library are:

    - icao24: the hexadecimal transponder identifier
    - registration: the tail number of the aircraft
    - typecode: the short identifier for the type of aircraft
    - model: the full name of the aircraft

    For example, you may download and uncompress Junzi's deprecated database
    from https://junzis.com/adb/ then edit the configuration file:

    .. parsed-literal::

        [aircraft]

        database = /home/xo/Downloads/aircraft_db/aircraft_db.csv

        icao24 = icao
        registration = regid
        typecode = mdl
        model = type

    """

    cache_dir: Path
    columns_options: ClassVar[dict[str, dict[str, Any]]] = dict(  # type: ignore
        icao24=dict(), registration=dict(), typecode=dict(), model=dict()
    )

    def __init__(self, data: None | pd.DataFrame = None) -> None:
        if data is None:
            self.data = self.opensky_db
        else:
            self.data = data
        other_columns = ["operator", "owner", "age"]
        for column in other_columns:
            if column in self.data.columns:
                self.columns_options[column] = dict(max_width=30)

    def download_junzis(self) -> None:  # coverage: ignore
        filename = self.cache_dir / "junzis_db.pkl"
        if filename.exists():
            self.data = pd.read_pickle(filename).fillna("")

        f = client.get("https://junzis.com/adb/download/aircraft_db.zip")
        with zipfile.ZipFile(io.BytesIO(f.content)) as zfile:
            with zfile.open("aircraft_db.csv", "r") as dbfile:
                self.data = (
                    pd.read_csv(dbfile, dtype=str)
                    .fillna("")
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
                self.data.to_pickle(self.cache_dir / "junzis_db.pkl")

    def download_opensky(self) -> None:  # coverage: ignore
        """Downloads the latest version of the OpenSky aircraft database.

        Reference: https://opensky-network.org/aircraft-database

        """

        _log.warning("Downloading OpenSky aircraft database")
        file_url = "https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv"
        f = client.get(file_url)
        total = int(f.headers["Content-Length"])
        buffer = io.BytesIO()
        for chunk in tqdm(
            f.iter_bytes(chunk_size=1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="download",
        ):
            buffer.write(chunk)

        buffer.seek(0)
        self.data = pd.read_csv(
            buffer,
            dtype={"icao24": str, "operator": str},
            skiprows=[1],
            engine="c",
            keep_default_na=False,
        )
        self.data.to_pickle(self.cache_dir / "opensky_db.pkl")

    @property
    def opensky_db(self) -> pd.DataFrame:
        if not (self.cache_dir / "opensky_db.pkl").exists():
            self.download_opensky()
        _log.info("Loading OpenSky aircraft database")
        return pd.read_pickle(self.cache_dir / "opensky_db.pkl")

    def __getitem__(self: T, name: str | list[str]) -> None | T:
        """Requests an aircraft by icao24 or registration (exact match)."""

        if isinstance(name, str):
            df = self.data.query(
                "icao24 == @name.lower() or registration == @name.upper()"
            )
        else:
            df = self.data.query("icao24 in @name or registration in @name")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def get_unique(self, name: str) -> None | Tail:
        _log.warn("Use .get() function instead", DeprecationWarning)
        return self.get(name)

    def get(self, name: str) -> None | Tail:
        """Returns information about an aircraft based on its icao24 or
        registration information as a dictionary. Only the first aircraft
        matching the pattern passed in parameter is returned.

        Country information are based on the icao24 identifier, category
        information are based on the registration information.

        :param name: the icao24 identifier or the tail number of the aircraft

        >>> from traffic.data import aircraft
        >>> aircraft.get("F-GFKY")
        Tail(icao24='391558', registration='F-GFKY', typecode='A320', flag='🇫🇷')
        """
        df = self[name]
        if df is None:
            return None
        return Tail({**dict(df.data.iloc[0]), **country(dict(df.data.iloc[0]))})

    def operator(self: T, name: str) -> None | T:
        """Requests an aircraft by owner or operator (fuzzy match).

        :param name: the owner or operator of the aircraft

        >>> aircraft.operator("British Antarctic")  # doctest: +SKIP
          icao24   registration   typecode   model   operator
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          4241f4   VP-FBB         DHC6               British Antarctic Survey
          4241f5   VP-FBL         DHC6               British Antarctic Survey
          4241f6   VP-FAZ         DHC6               British Antarctic Survey
          43be5b   VP-FBQ         DHC7               British Antarctic Survey

        """
        return self.query(f"operator.str.contains('{name}')")

    def stats(self: T, name: str) -> None | pd.DataFrame:
        """Computes stats of owned or operated aircraft (fuzzy match).

        :param name: the owner or operator of the aircraft

        >>> aircraft.stats("Air France")  # doctest: +SKIP
                                    model  icao24
        typecode
        A318              Airbus A318-111      16
        A319              Airbus A319-113      37
        A320              Airbus A320-214      48
        A321              Airbus A321-212      20
        A332              Airbus A330-203      15
        ...
        """
        subset = self.operator(name)
        if subset is None:
            return None
        return (
            subset.drop_duplicates("icao24")
            .groupby("typecode")
            .agg(dict(model="max", icao24="count"))
        )

    def model(self: T, name: str) -> None | T:
        """Requests an aircraft by model or typecode (fuzzy match).

        :param name: the model or the typecode of the aircraft

        >>> aircraft.model("A320")  # doctest: +SKIP
          icao24   registration   typecode   model         operator   owner
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          38005a   F-WWBA         A320       Airbus A320   Airbus     Airbus
          38009a   F-WWDA         A20N       Airbus A320   Airbus     Airbus
          38019a   F-WWIA         A20N       Airbus A320   Airbus     Airbus
          3801ba   F-WWIB         A20N       Airbus A320   Airbus     Airbus
          3801da   F-WWIC         A20N       Airbus A320   Airbus     Airbus
          3801fa   F-WWID         A20N       Airbus A320   Airbus     Airbus
          380d5a   F-WWBB         A20N       Airbus A320   Airbus     Airbus
          380d7a   F-WWBC         A20N       Airbus A320   Airbus     Airbus
          380d9a   F-WWBD         A20N       Airbus A320   Airbus     Airbus
          380dba   F-WWBF         A20N       Airbus A320   Airbus     Airbus
          ... (7021 more entries)

        """

        return self.query(
            f"model.str.contains('{name.upper()}') or "
            f"typecode.str.contains('{name.upper()}')"
        )

    def registration(self: T, name: str) -> None | T:
        """Requests an aircraft by registration (fuzzy match).

        :param name: the tail number of the aircraft
        """
        return self.query(f"registration.str.contains('{name}')")

    def query(
        self: T, query_str: str = "", *args: Any, **kwargs: Any
    ) -> None | T:
        """Combines several requests.

        :param query_str: (default: empty)
            if non empty, is passed to the :py:meth:`~pd.DataFrame.query` method

        :param kwargs: all keyword arguments correspond to the name of
            other methods which are chained if query_str is non empty

        >>> aircraft.query(registration="^F-ZB", model="EC45")  # doctest: +SKIP
          icao24   registration   typecode   model                     operator
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          3b780f   F-ZBQB         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b49   F-ZBQL         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b4a   F-ZBQK         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b50   F-ZBQI         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b51   F-ZBQH         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b52   F-ZBQG         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b8b   F-ZBQF         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b8c   F-ZBQF         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b8d   F-ZBQD         EC45       Airbus Helicopters H145   French Securite Civile
          3b7b8e   F-ZBQC         EC45       Airbus Helicopters H145   French Securite Civile
          ... (19 more entries)
        """
        if query_str != "":
            return super().query(query_str, *args, **kwargs)

        subs = (getattr(self, key)(value) for key, value in kwargs.items())
        res = reduce(
            lambda a, b: a.merge(b, how="inner"),
            (elt.data for elt in subs if elt is not None),
        )
        return self.__class__(res)
