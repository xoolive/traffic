import operator
import pickle
import re
import warnings
import zipfile
from collections import defaultdict
from functools import lru_cache, partial
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Set, Tuple)
from xml.etree import ElementTree

import numpy as np
from matplotlib.patches import Polygon as MplPolygon

import pyproj
from fastkml import kml
from fastkml.geometry import Geometry
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import cascaded_union, transform

from ..kml import toStyle  # type: ignore
from ..core.mixins import ShapelyMixin


class ExtrudedPolygon(NamedTuple):
    polygon: Polygon
    lower: float
    upper: float


class SectorInfo(NamedTuple):
    name: str
    type: Optional[str]


SectorList = List[ExtrudedPolygon]

components: Dict[str, Set[SectorInfo]] = defaultdict(set)

class Sector(ShapelyMixin):

    def __init__(self, name: str, elements: List[ExtrudedPolygon],
                 type_: Optional[str]=None) -> None:
        self.elements: List[ExtrudedPolygon] = elements
        self.name: str = name
        self.type: Optional[str] = type_

    def flatten(self) -> Polygon:
        return cascaded_union([p.polygon for p in self])

    @property
    def shape(self):
        return self.flatten()

    def __getitem__(self, *args) -> ExtrudedPolygon:
        return self.elements.__getitem__(*args)

    def __add__(self, other: 'Sector') -> 'Sector':
        union = cascaded_union_with_alt(list(self) + list(other))
        return Sector(f"{self.name}+{other.name}", union)

    def __iter__(self) -> Iterator[ExtrudedPolygon]:
        return self.elements.__iter__()

    def _repr_html_(self):
        title = f'<b>{self.name}/{self.type}</b>'
        shapes = ''
        title += '<ul>'
        for polygon in self:
            title += f'<li>{polygon.lower}, {polygon.upper}</li>'
            shapes += polygon.polygon._repr_svg_()
        title += '</ul>'
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(shapes)

    def __repr__(self):
        return f"Sector {self.name}/{self.type}"

    def __str__(self):
        return f"""Sector {self.name} with {len(self.elements)} parts"""

    def annotate(self, ax, **kwargs):
        if 'projection' in ax.__dict__:
            from cartopy.crs import PlateCarree
            kwargs['transform'] = PlateCarree()
        if 's' not in kwargs:
            kwargs['s'] = self.name
        ax.text(*np.array(self.centroid), **kwargs)

    def plot(self, ax, **kwargs):
        flat = self.flatten()
        if isinstance(flat, MultiPolygon):
            for poly in flat:
                # quick and dirty
                sub = Sector("", [ExtrudedPolygon(poly, 0, 0)])
                sub.plot(ax, **kwargs)
            return
        coords = np.stack(flat.exterior.coords)
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
    def components(self) -> Set[SectorInfo]:
        return components[self.name]

    def decompose(self, extr_p):
        c = np.stack(extr_p.polygon.exterior.coords)
        alt = np.zeros(c.shape[0], dtype=float)

        alt[:] = min(extr_p.upper, 400) * 30.48
        upper_layer = np.c_[c, alt]
        yield Polygon(upper_layer)
        alt[:] = max(0, extr_p.lower) * 30.48
        lower_layer = np.c_[c, alt][::-1, :]
        yield Polygon(lower_layer)

        for i, j in zip(range(c.shape[0]-1), range(c.shape[0], 1, -1)):
            yield Polygon(np.r_[lower_layer[i:i+2,:], upper_layer[j-2:j, :]])

    def export_kml(self, styleUrl:Optional[kml.StyleUrl]=None,
                   color:Optional[str]=None, alpha:float=.5):
        if color is not None:
            # the style will be set only if the kml.export context is open
            styleUrl = toStyle(color)
        folder = kml.Folder(name=self.name, description=self.type)
        for extr_p in self:
            for elt in self.decompose(extr_p):
                placemark = kml.Placemark(styleUrl=styleUrl)
                placemark.geometry = kml.Geometry(
                    geometry=elt, altitude_mode='relativeToGround')
                folder.append(placemark)
        return folder

    def export_json(self) -> Dict[str, Any]:
        export: Dict[str, Any] = {'name': self.name, 'type': self.type}
        shapes = []
        for p in self:
            shapes.append({'upper': p.upper,
                           'lower': p.lower,
                           'polygon': mapping(p.polygon)})
        export['shapes'] = shapes
        return export

def cascaded_union_with_alt(polyalt: SectorList) -> SectorList:
    altitudes = set(alt for _, *low_up in polyalt for alt in low_up)
    slices = sorted(altitudes)
    if len(slices) == 1 and slices[0] is None:
        simple_union = cascaded_union([p for p, *_ in polyalt])
        return [ExtrudedPolygon(simple_union, float("-inf"), float("inf"))]
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

        for filename in ['Airspace.BASELINE', 'DesignatedPoint.BASELINE',
                         'Navaid.BASELINE']:

            if ~(self.airac_path / filename).exists():
                zippath = zipfile.ZipFile(
                    self.airac_path.joinpath(f"{filename}.zip").as_posix())
                zippath.extractall(self.airac_path.as_posix())

        self.tree = ElementTree.parse(
            (self.airac_path / 'Airspace.BASELINE').as_posix())

        for airspace in self.tree.findall(
                'adrmsg:hasMember/aixm:Airspace', self.ns):

            identifier = airspace.find('gml:identifier', self.ns)
            assert(identifier is not None)
            assert(identifier.text is not None)
            self.full_dict[identifier.text] = airspace

        points = ElementTree.parse((self.airac_path / 'DesignatedPoint.BASELINE').
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

        points = ElementTree.parse((self.airac_path / 'Navaid.BASELINE').as_posix())

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

        self.initialized = True

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
        designator = airspace.find("aixm:designator", self.ns)
        if designator is not None:
            name = designator.text
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
                    new_t = ats.find("aixm:type", self.ns)
                    if new_d is not None:
                        if designator is not None:
                            components[name].add(SectorInfo(
                                new_d.text,
                                new_t.text if new_t is not None else None))
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

                    break  # only one timeslice

            if upper == float('inf') and lower == float('-inf'):
                polygons += cascaded_union_with_alt(block_poly)
            else:
                polygons.append(ExtrudedPolygon(cascaded_union(
                    [p for (p, *_) in block_poly]), lower, upper))

        return(cascaded_union_with_alt(polygons))

    def __getitem__(self, name: str) -> Optional[Sector]:
        return next(self.search(name, operator.eq), None)

    def search(self, name: str, cmp: Callable=re.match) -> Iterator[Sector]:

        if not self.initialized:
            self.init_cache()

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

                if (designator is not None and cmp(name, designator.text) and
                        (type_ is None or
                         ts.find("aixm:type", self.ns).text == type_)):

                    polygon = self.make_polygon(ts)
                    type_ = ts.find("aixm:type", self.ns).text
                    if len(polygon) > 0:
                        yield Sector(designator.text, polygon, type_)
                    else:
                        warnings.warn(
                            f"{designator.text} produces an empty sector",
                            RuntimeWarning)

    def parse(self, pattern: str, cmp=re.match):

        if not self.initialized:
            self.init_cache()

        name = pattern
        names = name.split('/')
        type_pattern: Optional[str] = None

        if len(names) > 1:
            name, type_pattern = names

        for airspace in self.tree.findall(
                        'adrmsg:hasMember/aixm:Airspace', self.ns):
            for ts in airspace.findall(
                    "aixm:timeSlice/aixm:AirspaceTimeSlice", self.ns):

                type_ = ts.find("aixm:type", self.ns)
                designator = ts.find('aixm:designator', self.ns)

                if ((type_pattern is None or
                     (type_ is not None and type_.text == type_pattern))
                        and (designator is not None and
                             cmp(name, designator.text))):
                    yield SectorInfo(
                        designator.text,
                        type_.text if type_ is not None else None)

