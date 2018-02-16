import pickle
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from xml.etree import ElementTree

import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import Polygon
from shapely.ops import cascaded_union


ExtrudedPolygon = NamedTuple('ExtrudedPolygon',
                             [('polygon', Polygon),
                              ('lower', float), ('upper', float)])
SectorList = List[ExtrudedPolygon]


class Sector(object):

    def __init__(self, name: str, area: List[ExtrudedPolygon],
                 type_: Optional[str]=None) -> None:
        self.area: List[ExtrudedPolygon] = area
        self.name: str = name
        self.type: Optional[str] = type_

    def flatten(self) -> Polygon:
        return cascaded_union([p.polygon for p in self])

    def intersects(self, structure):
        pass

    def __getitem__(self, *args):
        return self.area.__getitem__(*args)

    def __iter__(self):
        return self.area.__iter__()

    def _repr_svg_(self):
        for polygon in self:
            print(polygon.lower, polygon.upper)
        return self.flatten()._repr_svg_()

    def __repr__(self):
        return f"Sector {self.name}"

    def __str__(self):
        return f"""Sector {self.name} with {len(self.area)} parts"""

    def plot(self, ax, **kwargs):
        coords = np.stack(self.flatten().exterior.coords)
        if 'projection' in ax.__dict__:
            from cartopy.crs import PlateCarree
            coords = ax.projection.transform_points(
                PlateCarree(), *coords.T)[:, :2]
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'None'
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'red'
        ax.add_patch(MplPolygon(coords, **kwargs))

    @property
    def bounds(self) -> Tuple[float, ...]:
        return self.flatten().bounds

def cascaded_union_with_alt(polyalt: SectorList) -> SectorList:
    altitudes = set(alt for _, *low_up in polyalt for alt in low_up)
    slices = sorted(altitudes)
    results: List[ExtrudedPolygon] = []
    for low, up in zip(slices, slices[1:]):
        matched_poly = [p for (p, low_, up_) in polyalt
                        if low_ <= low <= up_ and low_ <= up <= up_]
        new_poly = ExtrudedPolygon(cascaded_union(matched_poly), low, up)
        if len(results) > 0 and new_poly.polygon.equals(results[-1].polygon):
            merged = ExtrudedPolygon(new_poly.polygon, results[-1].lower, up)
            results[-1] = merged
        else:
            results.append(new_poly)
    return results


class SectorParser(object):

    ns = {'adrmsg': 'http://www.eurocontrol.int/cfmu/b2b/ADRMessage',
          'aixm': 'http://www.aixm.aero/schema/5.1',
          'gml': 'http://www.opengis.net/gml/3.2',
          'xlink': 'http://www.w3.org/1999/xlink'}

    def __init__(self, airac_path: Path, cache_dir: Path) -> None:

        self.full_dict: Dict[str, Any] = {}
        self.all_points: Dict[str, Tuple[float, ...]] = {}

        assert airac_path.is_dir()

        cache_file = cache_dir / "airac.cache"
        if cache_file.exists():
            with cache_file.open("rb") as fh:
                self.full_dict, self.all_points, self.tree = pickle.load(fh)
                return

        for filename in ['Airspace.BASELINE', 'DesignatedPoint.BASELINE',
                         'Navaid.BASELINE']:

            if ~(airac_path / filename).exists():
                zippath = zipfile.ZipFile(
                    airac_path.joinpath(f"{filename}.zip").as_posix())
                zippath.extractall(airac_path.as_posix())

        self.tree = ElementTree.parse(
            (airac_path / 'Airspace.BASELINE').as_posix())

        for airspace in self.tree.findall(
                'adrmsg:hasMember/aixm:Airspace', self.ns):

            identifier = airspace.find('gml:identifier', self.ns)
            assert(identifier is not None)
            assert(identifier.text is not None)
            self.full_dict[identifier.text] = airspace

        points = ElementTree.parse((airac_path / 'DesignatedPoint.BASELINE').
                                   as_posix())

        for point in points.findall(
                "adrmsg:hasMember/aixm:DesignatedPoint", self.ns):

            identifier = point.find("gml:identifier", self.ns)
            assert(identifier is not None)
            assert(identifier.text is not None)

            floats = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/"
                "aixm:location/aixm:Point/gml:pos", self.ns)
            assert(floats is not None)
            assert(floats.text is not None)

            self.all_points[identifier.text] = tuple(
                float(x) for x in floats.text.split())

        points = ElementTree.parse((airac_path / 'Navaid.BASELINE').as_posix())

        for point in points.findall(
                "adrmsg:hasMember/aixm:Navaid", self.ns):

            identifier = point.find("gml:identifier", self.ns)
            assert(identifier is not None)
            assert(identifier.text is not None)

            floats = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/"
                "aixm:location/aixm:ElevatedPoint/gml:pos", self.ns)
            assert(floats is not None)
            assert(floats.text is not None)

            self.all_points[identifier.text] = tuple(
                float(x) for x in floats.text.split())

        with cache_file.open("wb") as fh:
            pickle.dump((self.full_dict, self.all_points, self.tree), fh)

    def append_coords(self, lr, block_poly):
        coords: List[Tuple[float, ...]] = []
        gml, xlink = self.ns['gml'], self.ns['xlink']
        for point in lr.iter():
            if point.tag in ('{%s}pos' % (gml),
                             '{%s}pointProperty' % (gml)):
                if point.tag.endswith('pos'):
                    coords.append(tuple(float(x) for x in point.text.split()))
                else:
                    points = point.attrib['{%s}href' % (xlink)]
                    coords.append(self.all_points[points.split(':')[2]])
        block_poly.append(
            (Polygon([(lon, lat) for lat, lon in coords]), None, None))


    @lru_cache(None)
    def make_polygon(self, airspace) -> SectorList:
        polygons: SectorList = []
        for block in airspace.findall(
                "aixm:geometryComponent/aixm:AirspaceGeometryComponent/"
                "aixm:theAirspaceVolume/aixm:AirspaceVolume", self.ns):
            block_poly: SectorList = []
            upper = block.find("aixm:upperLimit", self.ns)
            lower = block.find("aixm:lowerLimit", self.ns)

            upper = (float(upper.text) if upper is not None and
                     re.match("\d{3}", upper.text) else float("inf"))
            lower = (float(lower.text) if lower is not None and
                     re.match("\d{3}", lower.text) else float("-inf"))

            for component in block.findall(
                    "aixm:contributorAirspace/aixm:AirspaceVolumeDependency/"
                    "aixm:theAirspace", self.ns):
                key = component.attrib['{http://www.w3.org/1999/xlink}href']
                key = key.split(':')[2]
                child = self.full_dict[key]
                for ats in child.findall(
                        "aixm:timeSlice/aixm:AirspaceTimeSlice", self.ns):
                    new_d = ats.find("aixm:designator", self.ns)
                    if new_d is not None:
                        block_poly += self.make_polygon(ats)
                    else:
                        for sub in ats.findall(
                                "aixm:geometryComponent/"
                                "aixm:AirspaceGeometryComponent/"
                                "aixm:theAirspaceVolume/aixm:AirspaceVolume",
                                self.ns):

                            assert sub.find('aixm:lowerLimit', self.ns) is None

                            for lr in sub.findall(
                                    "aixm:horizontalProjection/aixm:Surface/"
                                    "gml:patches/gml:PolygonPatch/gml:exterior/"
                                    "gml:LinearRing", self.ns):
                                self.append_coords(lr, block_poly)

            if upper == float('inf') and lower == float('-inf'):
                polygons += cascaded_union_with_alt(block_poly)
            else:
                polygons.append(ExtrudedPolygon(cascaded_union(
                    [p for (p, *_) in block_poly]), lower, upper))

        return(cascaded_union_with_alt(polygons))

    def __getitem__(self, name: str) -> Sector:
        polygon = None
        type_: Optional[str] = None

        names = name.split('/')
        if len(names) > 1:
            name, type_ = names

        for airspace in self.tree.findall(
                'adrmsg:hasMember/aixm:Airspace', self.ns):
            for ts in airspace.findall(
                    "aixm:timeSlice/aixm:AirspaceTimeSlice", self.ns):

                designator = ts.find("aixm:designator", self.ns)

                if (designator is not None and
                        (designator.text == name) and
                        (type_ is None or
                         ts.find("aixm:type", self.ns).text == type_)):

                    polygon = self.make_polygon(ts)
                    type_ = ts.find("aixm:type", self.ns).text
                    break

        if polygon is None:
            raise ValueError(f"Sector {name} not found")

        return Sector(name, polygon, type_)
