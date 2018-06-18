from collections import defaultdict
from typing import (Any, Dict, Iterator, List, NamedTuple, Optional, Set,
                    Tuple, Union)

import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import Polygon, base, mapping
from shapely.ops import cascaded_union

from . import Flight, Traffic
from .mixins import ShapelyMixin


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
    def __init__(
        self,
        name: str,
        elements: List[ExtrudedPolygon],
        type_: Optional[str] = None,
    ) -> None:
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

    def __add__(self, other: "Sector") -> "Sector":
        union = cascaded_union_with_alt(list(self) + list(other))
        return Sector(f"{self.name}+{other.name}", union)

    def __iter__(self) -> Iterator[ExtrudedPolygon]:
        return self.elements.__iter__()

    def _repr_html_(self):
        title = f"<b>{self.name} ({self.type})</b>"
        shapes = ""
        title += "<ul>"
        for polygon in self:
            title += f"<li>{polygon.lower}, {polygon.upper}</li>"
            shapes += polygon.polygon._repr_svg_()
        title += "</ul>"
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(shapes)

    def __repr__(self):
        return f"Sector {self.name} ({self.type})"

    def __str__(self):
        return f"""Sector {self.name} with {len(self.elements)} parts"""

    def annotate(self, ax, **kwargs):
        if "projection" in ax.__dict__:
            from cartopy.crs import PlateCarree

            kwargs["transform"] = PlateCarree()
        if "s" not in kwargs:
            kwargs["s"] = self.name
        ax.text(*np.array(self.centroid), **kwargs)

    def plot(self, ax, **kwargs):
        flat = self.flatten()
        if isinstance(flat, base.BaseMultipartGeometry):
            for poly in flat:
                # quick and dirty
                sub = Sector("", [ExtrudedPolygon(poly, 0, 0)])
                sub.plot(ax, **kwargs)
            return

        if "facecolor" not in kwargs:
            kwargs["facecolor"] = "None"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "red"

        if "projection" in ax.__dict__:
            from cartopy.crs import PlateCarree
            ax.add_geometries([flat], crs=PlateCarree(), **kwargs)
        else:
            ax.add_patch(MplPolygon(list(flat.exterior.coords), **kwargs))

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

        for i, j in zip(range(c.shape[0] - 1), range(c.shape[0], 1, -1)):
            yield Polygon(
                np.r_[
                    lower_layer[i:i+2, :],  # noqa: E226
                    upper_layer[j-2:j, :],  # noqa: E226
                ]
            )

    def export_json(self) -> Dict[str, Any]:
        export: Dict[str, Any] = {"name": self.name, "type": self.type}
        shapes = []
        for p in self:
            shapes.append(
                {
                    "upper": p.upper,
                    "lower": p.lower,
                    "polygon": mapping(p.polygon),
                }
            )
        export["shapes"] = shapes
        return export


def cascaded_union_with_alt(polyalt: SectorList) -> SectorList:
    altitudes = set(alt for _, *low_up in polyalt for alt in low_up)
    slices = sorted(altitudes)
    if len(slices) == 1 and slices[0] is None:
        simple_union = cascaded_union([p for p, *_ in polyalt])
        return [ExtrudedPolygon(simple_union, float("-inf"), float("inf"))]
    results: List[ExtrudedPolygon] = []
    for low, up in zip(slices, slices[1:]):
        matched_poly = [
            p
            for (p, low_, up_) in polyalt
            if low_ <= low <= up_ and low_ <= up <= up_
        ]
        new_poly = ExtrudedPolygon(cascaded_union(matched_poly), low, up)
        if len(results) > 0 and new_poly.polygon.equals(results[-1].polygon):
            merged = ExtrudedPolygon(new_poly.polygon, results[-1].lower, up)
            results[-1] = merged
        else:
            results.append(new_poly)
    return results


# -- Methods below are placed here because of possible circular imports --


def _traffic_inside_bbox(
    traffic: Traffic, bounds: Union[Sector, Tuple[float, ...]]
) -> Traffic:

    if isinstance(bounds, Sector):
        bounds = bounds.flatten().bounds

    if isinstance(bounds, base.BaseGeometry):
        bounds = bounds.bounds

    west, south, east, north = bounds

    query = "{0} <= longitude <= {2} & {1} <= latitude <= {3}"
    query = query.format(west, south, east, north)

    data = traffic.data.query(query)

    return traffic.__class__(data)


def _traffic_intersects(traffic: Traffic, sector: Sector) -> Traffic:
    return Traffic.from_flights(
        flight for flight in traffic if flight.intersects(sector)
    )


def _flight_intersects(flight: Flight, sector: Sector) -> bool:
    for layer in sector:
        linestring = flight.airborne().linestring
        if linestring is None:
            return False
        ix = linestring.intersection(layer.polygon)
        if not ix.is_empty:
            if isinstance(ix, base.BaseMultipartGeometry):
                for part in ix:
                    if any(
                        100 * layer.lower < x[2] < 100 * layer.upper
                        for x in part.coords
                    ):
                        return True
            else:
                if any(
                    100 * layer.lower < x[2] < 100 * layer.upper
                    for x in ix.coords
                ):
                    return True
    return False


# -- The ugly monkey-patching --

setattr(Traffic, "inside_bbox", _traffic_inside_bbox)
setattr(Traffic, "intersects", _traffic_intersects)
setattr(Flight, "intersects", _flight_intersects)
