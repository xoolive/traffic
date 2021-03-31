# fmt: off

import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from shapely.geometry import base, polygon, shape

from ....core.airspace import (
    Airspace, AirspaceInfo, ExtrudedPolygon, components
)

# https://www.nm.eurocontrol.int/HELP/Airspaces.html

# fmt: on


def _re_match_ignorecase(x, y):
    return re.match(x, y, re.IGNORECASE)


class NMAirspaceParser(object):

    nm_path: Optional[Path] = None

    def __init__(self, config_file: Path) -> None:
        self.config_file = config_file

        self.polygons: Dict[str, base.BaseGeometry] = {}
        self.elements: Dict[str, List[Airspace]] = defaultdict(list)
        self.airspaces: Dict[str, List[str]] = defaultdict(list)
        self.description: Dict[str, str] = dict()
        self.types: Dict[str, str] = dict()
        self.initialized = False

    def update_path(self, path: Path):
        self.nm_path = path
        self.init_cache()

    def init_cache(self):
        msg = f"Edit file {self.config_file} with NM directory"

        if self.nm_path is None:
            raise RuntimeError(msg)

        are_file = next(self.nm_path.glob("Sectors_*.are"), None)
        if are_file is None:
            raise RuntimeError(f"No Sectors_*.are file found in {self.nm_path}")
        self.read_are(are_file)

        sls_file = next(self.nm_path.glob("Sectors_*.sls"), None)
        if sls_file is None:
            raise RuntimeError(f"No Sectors_*.sls file found in {self.nm_path}")
        self.read_sls(sls_file)

        spc_file = next(self.nm_path.glob("Sectors_*.spc"), None)
        if spc_file is None:
            raise RuntimeError(f"No Sectors_*.spc file found in {self.nm_path}")
        self.read_spc(spc_file)

        self.initialized = True

    def read_are(self, filename: Path) -> None:
        logging.info(f"Reading ARE file {filename}")
        nb_points = 0
        area_coords: List[Tuple[float, float]] = list()
        name = None
        with filename.open("r") as f:
            for line in f.readlines():
                if nb_points == 0:
                    if name is not None:
                        geometry = {
                            "type": "Polygon",
                            "coordinates": [area_coords],
                        }
                        self.polygons[name] = polygon.orient(
                            shape(geometry), -1
                        )

                    area_coords.clear()
                    nb, *_, name = line.split()
                    nb_points = int(nb)
                else:
                    lat, lon = line.split()
                    area_coords.append((float(lon) / 60, float(lat) / 60))
                    nb_points -= 1

            if name is not None:
                geometry = {"type": "Polygon", "coordinates": [area_coords]}
                self.polygons[name] = polygon.orient(shape(geometry), -1)

    def read_sls(self, filename: Path) -> None:
        logging.info(f"Reading SLS file {filename}")
        with filename.open("r") as fh:
            for line in fh.readlines():
                name, _, polygon, lower, upper = line.split()
                self.elements[name].append(
                    Airspace(
                        name,
                        [
                            ExtrudedPolygon(
                                self.polygons[polygon],
                                float(lower),
                                float(upper),
                            )
                        ],
                    )
                )

    def read_spc(self, filename: Path) -> None:
        logging.info(f"Reading SPC file {filename}")
        with open(filename, "r") as fh:
            for line in fh.readlines():
                letter, name, *after = line.split(";")
                if letter == "A":
                    cur = self.airspaces[name]
                    self.description[name] = after[0]
                    self.types[name] = after[1]
                elif letter == "S":
                    cur.append(name)
                    self.types[name] = after[0].strip()

    @lru_cache()
    def _ipython_key_completions_(self) -> Set[str]:
        return {*self.types.keys()}

    @lru_cache()
    def __getitem__(self, name: str) -> Optional[Airspace]:

        if not self.initialized:
            self.init_cache()

        list_names = self.airspaces.get(name, None)

        if list_names is None:
            elts = self.elements.get(name, None)
            if elts is None:
                return None
            else:
                airspace = sum(elts[1:], elts[0])
                airspace.name = self.description.get(name, name)
                airspace.type = self.types.get(name, "")
                return airspace

        list_airspaces: List[Airspace] = list()
        components_info: Set[AirspaceInfo] = set()
        for elt_name in list_names:
            element = self[elt_name]
            if element is not None:
                list_airspaces.append(element)
                components_info.add(
                    AirspaceInfo(elt_name, self.types[elt_name])
                )

        components[name] = components_info
        airspace = sum(list_airspaces[1:], list_airspaces[0])
        airspace.designator = name
        airspace.name = self.description.get(name, name)
        airspace.type = self.types[name]

        return airspace

    def parse(
        self, pattern: str, cmp: Callable = _re_match_ignorecase
    ) -> Iterator[AirspaceInfo]:

        if not self.initialized:
            self.init_cache()

        name = pattern
        names = name.split("/")
        type_pattern: Optional[str] = None

        if len(names) > 1:
            name, type_pattern = names

        for key, value in self.types.items():
            if cmp(name, key) and (
                type_pattern is None or value == type_pattern
            ):
                yield AirspaceInfo(key, value)

    def search(
        self, pattern: str, cmp: Callable = _re_match_ignorecase
    ) -> Iterator[Airspace]:

        if not self.initialized:
            self.init_cache()

        name = pattern
        names = name.split("/")
        type_pattern: Optional[str] = None

        if len(names) > 1:
            name, type_pattern = names

        for key, value in self.types.items():
            if cmp(name, key) and (
                type_pattern is None or value == type_pattern
            ):
                airspace = self[key]
                if airspace is not None:
                    yield airspace
