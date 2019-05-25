# flake8: noqa

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Tuple, Union

import pandas as pd
import requests

from ...core.mixins import DataFrameMixin, PointMixin, ShapelyMixin
from ...drawing import Nominatim, location


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


class Navaid(NavaidTuple, PointMixin):
    def __getattr__(self, name):
        if name == "latitude":
            return self.lat
        if name == "longitude":
            return self.lon
        if name == "altitude":
            return self.alt

    def __repr__(self):
        if self.type == "FIX":
            return f"{self.name} ({self.type}): {self.lat} {self.lon}"
        else:
            return (
                f"{self.name} ({self.type}): {self.lat} {self.lon}"
                f" {self.alt:.0f} "
                f"{self.description if self.description is not None else ''}"
                f" {self.frequency}{'kHz' if self.type=='NDB' else 'MHz'}"
            )


__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "xoolive/traffic/master/data/navdata"


class Navaids(DataFrameMixin):

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

    def __init__(self, data: Optional[pd.DataFrame] = None):
        self._data: Optional[pd.DataFrame] = data

    def download_data(self) -> None:  # coverage: ignore
        """Downloads the latest version of the navaid database from the
        repository.
        """

        navaids = []
        c = requests.get(f"{base_url}/earth_fix.dat")

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
                    None,
                    None,
                    None,
                    None,
                )
            )

        c = requests.get(f"{base_url}/earth_nav.dat")

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

        return self._data

    @lru_cache()
    def __getitem__(self, name: str) -> Optional[Navaid]:
        x = self.data.query(
            "description == @name.upper() or name == @name.upper()"
        )
        if x.shape[0] == 0:
            return None
        return Navaid(**dict(x.iloc[0]))

    def __iter__(self) -> Iterator[Navaid]:
        for _, x in self.data.iterrows():
            yield Navaid(**dict(x.iloc[0]))

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

    def extent(
        self,
        extent: Union[
            str, ShapelyMixin, Nominatim, Tuple[float, float, float, float]
        ],
    ) -> "Navaids":
        """
        Selects the subset of navigational beacons inside the given extent.

        The parameter extent may be passed as:

            - a string to query OSM Nominatim service;
            - the result of an OSM Nominatim query;
            - any kind of shape (including airspaces);
            - extents (west, east, south, north)

        >>> navaids['ZUE']
        ZUE (NDB): 30.9 20.06833333 0 ZUEITINA NDB 369.0kHz
        >>> navaids.extent('Switzerland')['ZUE']
        ZUE (VOR): 47.59216667 8.81766667 1730 ZURICH EAST VOR-DME 110.05MHz

        """
        if isinstance(extent, str):
            extent = location(extent)
        if isinstance(extent, ShapelyMixin):
            extent = extent.extent
        if isinstance(extent, Nominatim):
            extent = extent.extent

        west, east, south, north = extent

        return self.query(
            f"{south} <= lat <= {north} and {west} <= lon <= {east}"
        )
