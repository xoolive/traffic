# fmt: off

import logging
import operator
import pickle
import re
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
)
from xml.etree import ElementTree

from shapely.geometry import Polygon, polygon
from shapely.ops import cascaded_union

from ...core.airspace import components  # to be moved here TODO
from ...core.airspace import (
    Airspace, AirspaceInfo, AirspaceList,
    ExtrudedPolygon, cascaded_union_with_alt
)
from ...data.basic.airports import Airport

# fmt: on


class Point(NamedTuple):
    latitude: float
    longitude: float
    name: Optional[str]
    type: Optional[str]

    def __repr__(self) -> str:
        return f"{self.name} ({self.type}): {self.latitude} {self.longitude}"


class _Airport(NamedTuple):
    latitude: float
    longitude: float
    altitude: float
    iata: Optional[str]
    icao: Optional[str]
    name: Optional[str]
    city: Optional[str]
    type: Optional[str]

    def __repr__(self):
        return (
            f"{self.icao}/{self.iata}    {self.name} ({self.type})"
            f"\n\t{self.latitude} {self.longitude} altitude: {self.altitude}"
        )


def _re_match_ignorecase(x, y):
    return re.match(x, y, re.IGNORECASE)


class AIXMAirspaceParser(object):

    aixm_path: Optional[Path] = None
    cache_dir: Optional[Path] = None

    def __init__(self, config_file: Path) -> None:
        self.config_file = config_file
        self.initialized = False

    def init_cache(self) -> None:

        msg = f"Edit file {self.config_file} with AIXM directory"

        if self.aixm_path is None or self.cache_dir is None:
            raise RuntimeError(msg)

        self.full_dict: Dict[str, Any] = {}
        self.all_points: Dict[str, Point] = {}

        assert self.aixm_path.is_dir()
        self.ns: Dict[str, str] = dict()

        cache_file = self.cache_dir / "aixm.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as fh:
                try:
                    elts = pickle.load(fh)
                    self.full_dict = elts[0]
                    self.all_points = elts[1]
                    self.tree = elts[2]
                    self.ns = elts[3]

                    self.initialized = True
                    return
                except Exception:
                    logging.warning("aixm files: rebuilding cache file")

        for filename in [
            "AirportHeliport.BASELINE",
            "Airspace.BASELINE",
            "DesignatedPoint.BASELINE",
            "Navaid.BASELINE",
            "StandardInstrumentArrival.BASELINE",
        ]:

            if not (self.aixm_path / filename).exists():
                zippath = zipfile.ZipFile(
                    self.aixm_path.joinpath(f"{filename}.zip").as_posix()
                )
                zippath.extractall(self.aixm_path.as_posix())

        # The versions for namespaces may be incremented and make everything
        # fail just for that reason!
        for _, (key, value) in ElementTree.iterparse(
            (self.aixm_path / "Airspace.BASELINE").as_posix(),
            events=["start-ns"],
        ):
            self.ns[key] = value

        self.tree = ElementTree.parse(
            (self.aixm_path / "Airspace.BASELINE").as_posix()
        )

        for airspace in self.tree.findall(
            "adrmsg:hasMember/aixm:Airspace", self.ns
        ):

            identifier = airspace.find("gml:identifier", self.ns)
            assert identifier is not None
            assert identifier.text is not None
            self.full_dict[identifier.text] = airspace

        points = ElementTree.parse(
            (self.aixm_path / "DesignatedPoint.BASELINE").as_posix()
        )

        for point in points.findall(
            "adrmsg:hasMember/aixm:DesignatedPoint", self.ns
        ):

            identifier = point.find("gml:identifier", self.ns)
            assert identifier is not None
            assert identifier.text is not None

            floats = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/"
                "aixm:location/aixm:Point/gml:pos",
                self.ns,
            )
            assert floats is not None
            assert floats.text is not None

            designator = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/aixm:designator",
                self.ns,
            )
            type_ = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/aixm:type",
                self.ns,
            )

            name = designator.text if designator is not None else None
            type_str = type_.text if type_ is not None else None

            coords = tuple(float(x) for x in floats.text.split())
            self.all_points[identifier.text] = Point(
                coords[0], coords[1], name, type_str
            )

        points = ElementTree.parse(
            (self.aixm_path / "Navaid.BASELINE").as_posix()
        )

        for point in points.findall("adrmsg:hasMember/aixm:Navaid", self.ns):

            identifier = point.find("gml:identifier", self.ns)
            assert identifier is not None
            assert identifier.text is not None

            floats = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/"
                "aixm:location/aixm:ElevatedPoint/gml:pos",
                self.ns,
            )
            assert floats is not None
            assert floats.text is not None

            designator = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/aixm:designator", self.ns
            )
            type_ = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/aixm:type", self.ns
            )

            name = designator.text if designator is not None else None
            type_str = type_.text if type_ is not None else None

            coords = tuple(float(x) for x in floats.text.split())
            self.all_points[identifier.text] = Point(
                coords[0], coords[1], name, type_str
            )

        with cache_file.open("wb") as fh:
            pickle.dump(
                (self.full_dict, self.all_points, self.tree, self.ns), fh
            )

        self.initialized = True

    def points(self, name: str) -> Iterator[Point]:
        if not self.initialized:
            self.init_cache()
        return (
            point for point in self.all_points.values() if point.name == name
        )

    def append_coords(self, lr, block_poly):
        coords: List[Tuple[float, ...]] = []
        gml, xlink = self.ns["gml"], self.ns["xlink"]
        for point in lr.iter():
            if point.tag in ("{%s}pos" % (gml), "{%s}pointProperty" % (gml)):
                if point.tag.endswith("pos"):
                    coords.append(tuple(float(x) for x in point.text.split()))
                else:
                    points = point.attrib["{%s}href" % (xlink)]
                    current_point = self.all_points[points.split(":")[2]]
                    coords.append(
                        (current_point.latitude, current_point.longitude)
                    )
        block_poly.append(
            (
                polygon.orient(
                    Polygon([(lon, lat) for lat, lon in coords]), -1
                ),
                None,
                None,
            )
        )

    @lru_cache(None)
    def make_polygon(self, airspace) -> AirspaceList:
        polygons: AirspaceList = []
        designator = airspace.find("aixm:designator", self.ns)
        if designator is not None:
            name = designator.text
        for block in airspace.findall(
            "aixm:geometryComponent/aixm:AirspaceGeometryComponent/"
            "aixm:theAirspaceVolume/aixm:AirspaceVolume",
            self.ns,
        ):
            block_poly: AirspaceList = []
            upper = block.find("aixm:upperLimit", self.ns)
            lower = block.find("aixm:lowerLimit", self.ns)

            upper = (  # noqa: W605
                float(upper.text)
                if upper is not None and re.match(r"\d{3}", upper.text)
                else float("inf")
            )
            lower = (  # noqa: W605
                float(lower.text)
                if lower is not None and re.match(r"\d{3}", lower.text)
                else float("-inf")
            )

            for component in block.findall(
                "aixm:contributorAirspace/aixm:AirspaceVolumeDependency/"
                "aixm:theAirspace",
                self.ns,
            ):
                key = component.attrib["{http://www.w3.org/1999/xlink}href"]
                key = key.split(":")[2]
                child = self.full_dict[key]
                for ats in child.findall(
                    "aixm:timeSlice/aixm:AirspaceTimeSlice", self.ns
                ):
                    new_d = ats.find("aixm:designator", self.ns)
                    new_t = ats.find("aixm:type", self.ns)
                    if new_d is not None:
                        if designator is not None:
                            components[name].add(
                                AirspaceInfo(
                                    new_d.text,
                                    new_t.text if new_t is not None else None,
                                )
                            )
                        block_poly += self.make_polygon(ats)
                    else:
                        for sub in ats.findall(
                            "aixm:geometryComponent/"
                            "aixm:AirspaceGeometryComponent/"
                            "aixm:theAirspaceVolume/aixm:AirspaceVolume",
                            self.ns,
                        ):

                            assert sub.find("aixm:lowerLimit", self.ns) is None

                            for lr in sub.findall(
                                "aixm:horizontalProjection/aixm:Surface/"
                                "gml:patches/gml:PolygonPatch/gml:exterior/"
                                "gml:LinearRing",
                                self.ns,
                            ):
                                self.append_coords(lr, block_poly)

                    break  # only one timeslice

            if upper == float("inf") and lower == float("-inf"):
                polygons += cascaded_union_with_alt(block_poly)
            else:
                polygons.append(
                    ExtrudedPolygon(
                        cascaded_union([p for (p, *_) in block_poly]),
                        lower,
                        upper,
                    )
                )

        return cascaded_union_with_alt(polygons)

    def __getitem__(self, name: str) -> Optional[Airspace]:
        return next(self.search(name, operator.eq), None)

    def search(
        self, name: str, cmp: Callable = _re_match_ignorecase
    ) -> Iterator[Airspace]:

        if not self.initialized:
            self.init_cache()

        polygon = None
        type_: Optional[str] = None

        names = name.split("/")
        if len(names) > 1:
            name, type_ = names

        for airspace in self.tree.findall(
            "adrmsg:hasMember/aixm:Airspace", self.ns
        ):
            for ts in airspace.findall(
                "aixm:timeSlice/aixm:AirspaceTimeSlice", self.ns
            ):

                designator = ts.find("aixm:designator", self.ns)

                if (
                    designator is not None
                    and cmp(name, designator.text)
                    and (
                        type_ is None
                        or ts.find("aixm:type", self.ns).text == type_
                    )
                ):

                    polygon = self.make_polygon(ts)
                    type_ = ts.find("aixm:type", self.ns).text
                    name_ = ts.find("aixm:name", self.ns)
                    if len(polygon) > 0:
                        airspace = Airspace(
                            name=name_.text if name_ is not None else None,
                            elements=polygon,
                            type_=type_,
                            designator=designator.text
                            if designator is not None
                            else None,
                        )
                        yield airspace
                    else:
                        warnings.warn(
                            f"{designator.text} produces an empty airspace",
                            RuntimeWarning,
                        )

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

        for airspace in self.tree.findall(
            "adrmsg:hasMember/aixm:Airspace", self.ns
        ):
            for ts in airspace.findall(
                "aixm:timeSlice/aixm:AirspaceTimeSlice", self.ns
            ):

                type_ = ts.find("aixm:type", self.ns)
                designator = ts.find("aixm:designator", self.ns)

                if (
                    type_pattern is None
                    or (type_ is not None and type_.text == type_pattern)
                ) and (designator is not None and cmp(name, designator.text)):
                    yield AirspaceInfo(
                        designator.text,
                        type_.text if type_ is not None else None,
                    )

    def airports(self) -> Iterator[Tuple[str, _Airport]]:

        assert self.aixm_path is not None

        a_tree = ElementTree.parse(
            (self.aixm_path / "AirportHeliport.BASELINE").as_posix()
        )

        for elt in a_tree.findall(
            "adrmsg:hasMember/aixm:AirportHeliport", self.ns
        ):

            identifier = elt.find("gml:identifier", self.ns)
            assert identifier is not None
            assert identifier.text is not None

            apt = elt.find(
                "aixm:timeSlice/aixm:AirportHeliportTimeSlice", self.ns
            )
            if apt is None:
                continue

            nameElt = apt.find("aixm:name", self.ns)
            icaoElt = apt.find("aixm:locationIndicatorICAO", self.ns)
            iataElt = apt.find("aixm:designatorIATA", self.ns)
            typeElt = apt.find("aixm:controlType", self.ns)
            cityElt = apt.find("aixm:servedCity/aixm:City/aixm:name", self.ns)
            posElt = apt.find("aixm:ARP/aixm:ElevatedPoint/gml:pos", self.ns)
            altElt = apt.find(
                "aixm:ARP/aixm:ElevatedPoint/aixm:elevation", self.ns
            )

            if (
                posElt is None
                or posElt.text is None
                or altElt is None
                or altElt.text is None
                or icaoElt is None
            ):
                continue

            coords = tuple(float(x) for x in posElt.text.split())

            yield (
                identifier.text,
                _Airport(
                    coords[0],
                    coords[1],
                    float(altElt.text),
                    iataElt.text if iataElt is not None else None,
                    icaoElt.text,
                    nameElt.text if nameElt is not None else None,
                    cityElt.text if cityElt is not None else None,
                    typeElt.text if typeElt is not None else None,
                ),
            )

    def star(self, airport: Union[str, Airport]) -> Iterator[Tuple[Point, ...]]:

        if not self.initialized:
            self.init_cache()

        assert self.aixm_path is not None

        if isinstance(airport, Airport):
            airport = airport.icao

        airports_dict: Dict[str, _Airport] = {
            key: airport for key, airport in self.airports()
        }

        tree = ElementTree.parse(
            (self.aixm_path / "StandardInstrumentArrival.BASELINE").as_posix()
        )

        for elt in tree.findall(
            "adrmsg:hasMember/" "aixm:StandardInstrumentArrival", self.ns
        ):

            identifier = elt.find("gml:identifier", self.ns)
            assert identifier is not None
            assert identifier.text is not None

            star = elt.find(
                "aixm:timeSlice/" "aixm:StandardInstrumentArrivalTimeSlice",
                self.ns,
            )

            if star is None:
                logging.warning(
                    "No aixm:StandardInstrumentArrivalTimeSlice found"
                )
                continue

            handle = star.find("aixm:airportHeliport", self.ns)
            if handle is None:
                logging.warning("No aixm:airportHeliport found")
                continue

            airport_id = handle.attrib["{%s}href" % (self.ns["xlink"])]
            if airports_dict[airport_id.split(":")[2]].icao == airport:
                points = star.find(
                    "aixm:extension/"
                    "adrext:StandardInstrumentArrivalExtension",
                    self.ns,
                )

                if points is None:
                    logging.warning(
                        "No aixm:StandardInstrumentArrivalExtension found"
                    )
                    continue

                segment: List[Point] = []
                for p in points.getchildren():
                    x = p.find(
                        "aixm:TerminalSegmentPoint/"
                        "aixm:pointChoice_fixDesignatedPoint",
                        self.ns,
                    )

                    if x is not None:
                        x_id = x.attrib["{%s}href" % (self.ns["xlink"])]
                        point = self.all_points[x_id.split(":")[2]]
                        segment.append(point._replace(type=p.tag.split("}")[1]))

                    x = p.find(
                        "aixm:TerminalSegmentPoint/"
                        "aixm:pointChoice_navaidSystem",
                        self.ns,
                    )
                    if x is not None:
                        x_id = x.attrib["{%s}href" % (self.ns["xlink"])]
                        point = self.all_points[x_id.split(":")[2]]
                        segment.append(point._replace(type=p.tag.split("}")[1]))

                yield tuple(segment)
