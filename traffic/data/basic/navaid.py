# flake8: noqa

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

import pandas as pd

from ...core.mixins import GeoDBMixin, PointMixin, ShapelyMixin
from ...core.structure import Navaid, NavaidTuple
from ...drawing import Nominatim, location

__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "xoolive/traffic/master/data/navdata"


class Navaids(GeoDBMixin):

    """
    `VOR <https://en.wikipedia.org/wiki/VHF_omnidirectional_range>`_, `DME
    <https://en.wikipedia.org/wiki/Distance_measuring_equipment>`_ and `NDB
    <https://en.wikipedia.org/wiki/Non-directional_beacon>`_ are short-range
    radio navigation systems for aircraft.  They help aircraft with a receiving
    unit to determine their position and stay on course.

    The first airways were designed after the locations of these beacons. Recent
    progress in GNSS systems helped define more positions by their latitudes and
    longitudes, referred to as FIX.

    - Read more `here <https://aerosavvy.com/navigation-name-nonsense/>`_ about
      navigational beacons and how FIX names are decided.

    - Read more `here <scenarios/calibration.html>`_ about calibration of such
      equipment.

    A (deprecated) database of world navigational beacons is available as:

    >>> from traffic.data import navaids

    Any navigational beacon can be accessed by the bracket notation:

    >>> navaids['NARAK']
    NARAK (FIX): 44.29527778 1.74888889

    """

    cache_dir: Path
    alternatives: Dict[str, "Navaids"] = dict()
    name: str = "default"

    def __init__(self, data: Optional[pd.DataFrame] = None):
        self._data: Optional[pd.DataFrame] = data

    def __new__(cls, data: Optional[pd.DataFrame] = None) -> "Navaids":
        instance = super().__new__(cls)
        if instance.available:
            Navaids.alternatives[cls.name] = instance
        return instance

    @property
    def available(self) -> bool:
        return True

    def download_data(self) -> None:  # coverage: ignore
        """Downloads the latest version of the navaid database from the
        repository.
        """

        from .. import session

        navaids = []
        c = session.get(f"{base_url}/earth_fix.dat")

        for line in c.iter_lines():

            line = line.decode(encoding="ascii", errors="ignore").strip()

            # Skip empty lines or comments
            if len(line) < 3 or line[0] == "#":
                continue

            # Start with valid 2 digit latitude -45. or 52.
            if not ((line[0] == "-" and line[3] == ".") or line[2] == "."):
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example line:
            #  30.580372 -094.384169 FAREL
            fields = line.split()
            navaids.append(
                Navaid(
                    fields[2],
                    "FIX",
                    float(fields[0]),
                    float(fields[1]),
                    float("nan"),
                    None,
                    None,
                    None,
                )
            )

        c = session.get(f"{base_url}/earth_nav.dat")

        for line in c.iter_lines():

            line = line.decode(encoding="ascii", errors="ignore").strip()

            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example lines:
            # 2  58.61466599  125.42666626 451   522  30  0.0 A   Aldan NDB
            # 3  31.26894444 -085.72630556 334 11120  40 -3.0 OZR CAIRNS VOR-DME
            # type    lat       lon        elev freq  ?   var id   desc
            #   0      1         2           3    4   5    6   7    8

            fields = line.split()

            # Valid line starts with integers
            if not fields[0].isdigit():
                continue  # Next line

            # Get code for type of navaid
            itype = int(fields[0])

            # Type names
            wptypedict = {
                2: "NDB",
                3: "VOR",
                4: "ILS",
                5: "LOC",
                6: "GS",
                7: "OM",
                8: "MM",
                9: "IM",
                12: "DME",
                13: "TACAN",
            }

            # Type code never larger than 20
            if itype not in list(wptypedict.keys()):
                continue  # Next line

            wptype = wptypedict[itype]

            # Select types to read
            if wptype not in ["NDB", "VOR", "ILS", "GS", "DME", "TACAN"]:
                continue  # Next line

            # Find description
            try:
                idesc = line.index(fields[7]) + len(fields[7])
                description: Optional[str] = line[idesc:].strip().upper()
            except Exception:
                description = None

            navaids.append(
                Navaid(
                    fields[7],
                    wptype,
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3][1:])
                    if fields[3].startswith("0-")
                    else float(fields[3]),
                    float(fields[4])
                    if wptype == "NDB"
                    else float(fields[4]) / 100,
                    float(fields[6])
                    if wptype in ["VOR", "NDB", "ILS", "GS"]
                    else None,
                    description,
                )
            )

        self._data = pd.DataFrame.from_records(
            navaids, columns=NavaidTuple._fields
        )

        self._data.to_pickle(self.cache_dir / "traffic_navaid.pkl")

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        if not (self.cache_dir / "traffic_navaid.pkl").exists():
            self.download_data()
        else:
            logging.info("Loading navaid database")
            self._data = pd.read_pickle(self.cache_dir / "traffic_navaid.pkl")

        if self._data is not None:
            self._data = self._data.rename(
                columns=dict(alt="altitude", lat="latitude", lon="longitude")
            )
        return self._data

    @lru_cache()
    def __getitem__(self, name: str) -> Optional[Navaid]:
        x = self.data.query(
            "description == @name.upper() or name == @name.upper()"
        )
        if x.shape[0] == 0:
            return None
        dic = dict(x.iloc[0])
        if "altitude" not in dic:
            dic["altitude"] = None
            dic["frequency"] = None
            dic["magnetic_variation"] = None
        return Navaid(**dic)

    def global_get(self, name) -> Optional[Navaid]:
        """Search for a navaid from all alternative data sources."""
        for _key, value in self.alternatives.items():
            alt = value[name]
            if alt is not None:
                return alt
        return None

    def __iter__(self) -> Iterator[Navaid]:
        for _, x in self.data.iterrows():
            yield Navaid(**x)

    def search(self, name: str) -> "Navaids":
        """
        Selects the subset of airways matching name in the name or description
        field.

        .. warning::
            The same name may match several navigational beacons in the world.

        >>> navaids.search("ZUE")
                name  type  lat       lon       alt     frequency     description
        272107  ZUE   NDB   30.900000 20.068333 0.0     369.00  0.0   ZUEITINA NDB
        275948  ZUE   VOR   47.592167 8.817667  1730.0  110.05  2.0   ZURICH EAST VOR-DME
        290686  ZUE   DME   47.592167 8.817667  1730.0  110.05  NaN   ZURICH EAST VOR-DME
        """
        return self.__class__(
            self.data.query(
                "description == @name.upper() or name == @name.upper()"
            )
        )
