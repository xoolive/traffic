import operator
import pickle
import re
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree

from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from ...core.sector import components  # to be moved here TODO
from ...core.sector import (
    ExtrudedPolygon,
    Sector,
    SectorInfo,
    SectorList,
    cascaded_union_with_alt,
)


class SectorParser(object):

    ns = {
        "adrmsg": "http://www.eurocontrol.int/cfmu/b2b/ADRMessage",
        "aixm": "http://www.aixm.aero/schema/5.1",
        "gml": "http://www.opengis.net/gml/3.2",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    airac_path: Optional[Path] = None
    cache_dir: Optional[Path] = None

    def __init__(self, config_file: Path) -> None:
        self.config_file = config_file
        self.initialized = False

    def init_cache(self) -> None:

        msg = f"Edit file {self.config_file} with AIRAC directory"

        if self.airac_path is None or self.cache_dir is None:
            raise RuntimeError(msg)

        self.full_dict: Dict[str, Any] = {}
        self.all_points: Dict[str, Tuple[float, ...]] = {}

        assert self.airac_path.is_dir()

        cache_file = self.cache_dir / "airac.cache"
        if cache_file.exists():
            with cache_file.open("rb") as fh:
                self.full_dict, self.all_points, self.tree = pickle.load(fh)
                self.initialized = True
                return

        for filename in [
            "Airspace.BASELINE",
            "DesignatedPoint.BASELINE",
            "Navaid.BASELINE",
        ]:

            if ~(self.airac_path / filename).exists():
                zippath = zipfile.ZipFile(
                    self.airac_path.joinpath(f"{filename}.zip").as_posix()
                )
                zippath.extractall(self.airac_path.as_posix())

        self.tree = ElementTree.parse(
            (self.airac_path / "Airspace.BASELINE").as_posix()
        )

        for airspace in self.tree.findall(
            "adrmsg:hasMember/aixm:Airspace", self.ns
        ):

            identifier = airspace.find("gml:identifier", self.ns)
            assert identifier is not None
            assert identifier.text is not None
            self.full_dict[identifier.text] = airspace

        points = ElementTree.parse(
            (self.airac_path / "DesignatedPoint.BASELINE").as_posix()
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

            self.all_points[identifier.text] = tuple(
                float(x) for x in floats.text.split()
            )

        points = ElementTree.parse(
            (self.airac_path / "Navaid.BASELINE").as_posix()
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

            self.all_points[identifier.text] = tuple(
                float(x) for x in floats.text.split()
            )

        with cache_file.open("wb") as fh:
            pickle.dump((self.full_dict, self.all_points, self.tree), fh)

        self.initialized = True

    def append_coords(self, lr, block_poly):
        coords: List[Tuple[float, ...]] = []
        gml, xlink = self.ns["gml"], self.ns["xlink"]
        for point in lr.iter():
            if point.tag in ("{%s}pos" % (gml), "{%s}pointProperty" % (gml)):
                if point.tag.endswith("pos"):
                    coords.append(tuple(float(x) for x in point.text.split()))
                else:
                    points = point.attrib["{%s}href" % (xlink)]
                    coords.append(self.all_points[points.split(":")[2]])
        block_poly.append(
            (Polygon([(lon, lat) for lat, lon in coords]), None, None)
        )

    @lru_cache(None)
    def make_polygon(self, airspace) -> SectorList:
        polygons: SectorList = []
        designator = airspace.find("aixm:designator", self.ns)
        if designator is not None:
            name = designator.text
        for block in airspace.findall(
            "aixm:geometryComponent/aixm:AirspaceGeometryComponent/"
            "aixm:theAirspaceVolume/aixm:AirspaceVolume",
            self.ns,
        ):
            block_poly: SectorList = []
            upper = block.find("aixm:upperLimit", self.ns)
            lower = block.find("aixm:lowerLimit", self.ns)

            upper = (
                float(upper.text)
                if upper is not None and re.match("\d{3}", upper.text)
                else float("inf")
            )
            lower = (
                float(lower.text)
                if lower is not None and re.match("\d{3}", lower.text)
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
                                SectorInfo(
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

    def __getitem__(self, name: str) -> Optional[Sector]:
        return next(self.search(name, operator.eq), None)

    def search(self, name: str, cmp: Callable = re.match) -> Iterator[Sector]:

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
                    if len(polygon) > 0:
                        yield Sector(designator.text, polygon, type_)
                    else:
                        warnings.warn(
                            f"{designator.text} produces an empty sector",
                            RuntimeWarning,
                        )

    def parse(self, pattern: str, cmp=re.match):

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
                    yield SectorInfo(
                        designator.text,
                        type_.text if type_ is not None else None,
                    )
