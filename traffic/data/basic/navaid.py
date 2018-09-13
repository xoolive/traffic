import pickle
import re
from pathlib import Path
from typing import List, NamedTuple, Optional

import requests

from ...core.mixins import PointMixin


class NavaidTuple(NamedTuple):

    name: str
    type: str
    lat: float
    lon: float
    alt: Optional[float]
    frequency: Optional[float]
    magnetic_variation: Optional[float]
    description: Optional[str]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, name):
        if name == "latitude":
            return self.lat
        if name == "longitude":
            return self.lon
        if name == "altitude":
            return self.alt


class Navaid(NavaidTuple, PointMixin):
    def __repr__(self):
        if self.type == "FIX":
            return f"{self.name} ({self.type}): {self.lat} {self.lon}"
        else:
            return (
                f"{self.name} ({self.type}): {self.lat} {self.lon}"
                f" {self.alt:.0f}\n"
                f"{self.description if self.description is not None else ''}"
                f" {self.frequency}{'kHz' if self.type=='NDB' else 'MHz'}"
            )


__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "ProfHoekstra/bluesky/master/data/navdata"


class NavaidParser(object):

    cache: Optional[Path] = None

    def __init__(self) -> None:
        if self.cache is not None and self.cache.exists():
            with self.cache.open("rb") as fh:
                self.navaids = pickle.load(fh)
        else:
            self.initialize()
            if self.cache is not None:
                with self.cache.open("wb") as fh:
                    pickle.dump(self.navaids, fh)

    def __getitem__(self, name: str) -> Optional[Navaid]:
        return next(
            (pt for pt in self.navaids if (pt.name == name.upper())), None
        )

    def search(self, name: str) -> List[Navaid]:
        return list(
            (
                pt
                for pt in self.navaids
                if (
                    pt.description is not None
                    and (re.match(name, pt.description, re.IGNORECASE))
                )
                or (pt.name == name.upper())
            )
        )

    def initialize(self):
        self.navaids = []
        c = requests.get(f"{base_url}/fix.dat")

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
            self.navaids.append(
                Navaid(
                    fields[2],
                    "FIX",
                    float(fields[0]),
                    float(fields[1]),
                    None,
                    None,
                    None,
                    None,
                )
            )

        c = requests.get(f"{base_url}/nav.dat")

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

            # Valid line starst with integers
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
            if wptype not in ["NDB", "VOR", "DME", "TACAN"]:
                continue  # Next line

            # Find description
            try:
                idesc = line.index(fields[7]) + len(fields[7])
                description = line[idesc:].strip().upper()
            except Exception:
                description = None

            self.navaids.append(
                Navaid(
                    fields[7],
                    wptype,
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4])
                    if wptype == "NDB"
                    else float(fields[4]) / 100,
                    float(fields[6]) if wptype in ["VOR", "NDB"] else None,
                    description,
                )
            )
