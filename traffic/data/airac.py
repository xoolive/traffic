import os
import re
import pickle
import zipfile
import tempfile

from pathlib import Path
from functools import lru_cache
from xml.etree import ElementTree

import numpy as np

from shapely.ops import cascaded_union
from shapely.geometry import Polygon

from matplotlib.patches import Polygon as MplPolygon

full_dict = {}
all_points = {}
tree = None

ns = {'adrmsg': 'http://www.eurocontrol.int/cfmu/b2b/ADRMessage',
      'aixm': 'http://www.aixm.aero/schema/5.1',
      'gml': 'http://www.opengis.net/gml/3.2',
      'xlink': 'http://www.w3.org/1999/xlink'}


class Sector(object):

    def __init__(self, name, area):
        self.area = area
        self.name = name

    def flatten(self):
        return cascaded_union([p[0] for p in self])

    def intersects(self, structure):
        pass

    def __getitem__(self, *args):
        return self.area.__getitem__(*args)

    def _repr_svg_(self):
        for p, low, up in self:
            print(low, up)
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



def init_airac(airac_path, cache_dir):
    global full_dict, all_points, tree

    airac_path = Path(airac_path)
    assert airac_path.is_dir()

    cache_file = Path(cache_dir) / "airac.cache"
    if cache_file.exists():
        with cache_file.open("rb") as fh:
            full_dict, all_points, tree = pickle.load(fh)
            return

    for filename in ['Airspace.BASELINE', 'DesignatedPoint.BASELINE',
                     'Navaid.BASELINE']:

        if ~(airac_path / filename).exists():
            zippath = zipfile.ZipFile(airac_path.joinpath(f"{filename}.zip"))
            zippath.extractall(airac_path.as_posix())

    tree = ElementTree.parse((airac_path / 'Airspace.BASELINE').as_posix())


    for airspace in tree.findall('adrmsg:hasMember/aixm:Airspace', ns):
        full_dict[airspace.find('gml:identifier', ns).text] = airspace


    points = ElementTree.parse((airac_path / 'DesignatedPoint.BASELINE').
                               as_posix())

    for point in points.findall("adrmsg:hasMember/aixm:DesignatedPoint", ns):
        all_points[point.find("gml:identifier", ns).text] = tuple(
            float(x) for x in point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/"
                "aixm:location/aixm:Point/gml:pos", ns).text.split())

    points = ElementTree.parse((airac_path / 'Navaid.BASELINE').as_posix())

    for point in points.findall("adrmsg:hasMember/aixm:Navaid", ns):
        all_points[point.find("gml:identifier", ns).text] = tuple(
            float(x) for x in point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/"
                "aixm:location/aixm:ElevatedPoint/gml:pos", ns).text.split())

    with cache_file.open("wb") as fh:
        pickle.dump((full_dict, all_points, tree), fh)


def cascaded_union_with_alt(polyalt):
    altitudes = set(alt for _, *low_up in polyalt for alt in low_up)
    slices = sorted(altitudes)
    results = []
    last_matched = {}
    for low, up in zip(slices, slices[1:]):
        matched_poly = [p for (p, low_, up_) in polyalt
                        if low_ <= low <= up_ and low_ <= up <= up_]
        results.append([cascaded_union(matched_poly), low, up])
    return results

@lru_cache(None)
def make_polygon(airspace):
    polygons = []
    for block in airspace.findall(
        "aixm:geometryComponent/aixm:AirspaceGeometryComponent/"
        "aixm:theAirspaceVolume/aixm:AirspaceVolume", ns):
        block_poly = []
        upper = block.find("aixm:upperLimit", ns)
        lower = block.find("aixm:lowerLimit", ns)

        upper = (float(upper.text) if upper is not None and
                 re.match("\d{3}", upper.text) else float("inf"))
        lower = (float(lower.text) if lower is not None and
                 re.match("\d{3}", lower.text) else float("-inf"))

        for component in block.findall(
            "aixm:contributorAirspace/aixm:AirspaceVolumeDependency/"
            "aixm:theAirspace", ns):
            key = component.attrib['{http://www.w3.org/1999/xlink}href']
            key = key.split(':')[2]
            child = full_dict[key]
            for ats in child.findall(
                    "aixm:timeSlice/aixm:AirspaceTimeSlice", ns):
                new_d = ats.find("aixm:designator", ns)
                if new_d is not None:
                    block_poly += make_polygon(ats)
                else:
                    for sub in ats.findall(
                            "aixm:geometryComponent/"
                            "aixm:AirspaceGeometryComponent/"
                            "aixm:theAirspaceVolume/aixm:AirspaceVolume", ns):

                        assert sub.find('aixm:lowerLimit', ns) is None

                        for lr in sub.findall(
                                "aixm:horizontalProjection/aixm:Surface/"
                                "gml:patches/gml:PolygonPatch/gml:exterior/"
                                "gml:LinearRing", ns):
                            coords = []
                            for point in lr.iter():
                                if point.tag in ('{%s}pos' % (ns['gml']),
                                                 '{%s}pointProperty' % (ns['gml'])):
                                    if point.tag.endswith('pos'):
                                        coords.append([float(x) for x in point.text.split()])
                                    else:
                                        points = point.attrib['{http://www.w3.org/1999/xlink}href']
                                        coords.append(all_points[points.split(':')[2]])
                            block_poly.append((Polygon([(lon, lat) for lat, lon in coords]), None, None))

        if upper == float('inf') and lower == float('-inf'):
            polygons += cascaded_union_with_alt(block_poly)
        else:
            polygons.append((cascaded_union(
                [p for (p, *_) in block_poly]), lower, upper))

    return(cascaded_union_with_alt(polygons))


def get_area(name, type_ = None):
    polygon = None

    for airspace in tree.findall('adrmsg:hasMember/aixm:Airspace', ns):
        for ts in airspace.findall("aixm:timeSlice/aixm:AirspaceTimeSlice", ns):

            designator = ts.find("aixm:designator", ns)

            if (designator is not None and
                (designator.text == name) and
                (type_ is None or ts.find("aixm:type", ns).text == type_)):

                polygon = make_polygon(ts)
                break

    return Sector(name, polygon)
