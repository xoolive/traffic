# fmt: off

import json
from collections import defaultdict
from pathlib import Path
from typing import (Any, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple,
                    TypeVar, Union)

import numpy as np
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, base, mapping, shape
from shapely.ops import cascaded_union

from . import Flight, Traffic
from .lazy import lazy_evaluation
from .mixins import GeographyMixin, PointMixin, ShapelyMixin  # noqa: F401

# fmt: on


class ExtrudedPolygon(NamedTuple):
    polygon: Polygon
    lower: float
    upper: float


class AirspaceInfo(NamedTuple):
    name: str
    type: Optional[str]


AirspaceList = List[ExtrudedPolygon]
components: Dict[str, Set[AirspaceInfo]] = defaultdict(set)


class Airspace(ShapelyMixin):
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
        """Returns the 2D footprint of the airspace."""
        return cascaded_union([p.polygon for p in self])

    @property
    def shape(self):
        return self.flatten()

    def __getitem__(self, *args) -> ExtrudedPolygon:
        return self.elements.__getitem__(*args)

    def __add__(self, other: "Airspace") -> "Airspace":
        if other == 0:
            # useful for compatibility with sum() function
            return self
        union = cascaded_union_with_alt(list(self) + list(other))
        return Airspace(f"{self.name}+{other.name}", union)

    def __radd__(self, other):
        return self + other

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
        return f"Airspace {self.name} ({self.type})"

    def __str__(self):
        return f"""Airspace {self.name} with {len(self.elements)} parts"""

    def annotate(
        self, ax: GeoAxesSubplot, **kwargs
    ) -> None:  # coverage: ignore
        if "projection" in ax.__dict__:
            kwargs["transform"] = PlateCarree()
        if "s" not in kwargs:
            kwargs["s"] = self.name
        ax.text(*np.array(self.centroid), **kwargs)

    def plot(self, ax: GeoAxesSubplot, **kwargs) -> None:  # coverage: ignore
        flat = self.flatten()
        if isinstance(flat, base.BaseMultipartGeometry):
            for poly in flat:
                # quick and dirty
                sub = Airspace("", [ExtrudedPolygon(poly, 0, 0)])
                sub.plot(ax, **kwargs)
            return

        if "facecolor" not in kwargs:
            kwargs["facecolor"] = "None"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = ax._get_lines.get_next_color()

        if "projection" in ax.__dict__:
            ax.add_geometries([flat], crs=PlateCarree(), **kwargs)
        else:
            ax.add_patch(MplPolygon(list(flat.exterior.coords), **kwargs))

    @property
    def point(self) -> PointMixin:
        p = PointMixin()
        p.longitude, p.latitude = list(self.centroid.coords)[0]
        return p

    @property
    def components(self) -> Set[AirspaceInfo]:
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
                    lower_layer[i : i + 2, :],  # noqa: E203
                    upper_layer[j - 2 : j, :],  # noqa: E203
                ]
            )

    def above(self, level: int) -> "Airspace":
        return Airspace(
            self.name,
            list(c for c in self.elements if c.upper >= level),
            type_=self.type,
        )

    def below(self, level: int) -> "Airspace":
        return Airspace(
            self.name,
            list(c for c in self.elements if c.lower <= level),
            type_=self.type,
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

    @classmethod
    def from_json(cls, json: Dict[str, Any]):
        return cls(
            name=json["name"],
            type_=json["type"],
            elements=[
                ExtrudedPolygon(
                    polygon=shape(layer["polygon"]),
                    upper=layer["upper"],
                    lower=layer["lower"],
                )
                for layer in json["shapes"]
            ],
        )

    @classmethod
    def from_file(cls, filename: Union[Path, str]):
        path = Path(filename)
        with path.open("r") as fh:
            return cls.from_json(json.load(fh))


def cascaded_union_with_alt(polyalt: AirspaceList) -> AirspaceList:
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


T = TypeVar("T", bound="GeographyMixin")


def inside_bbox(
    geography: T,
    bounds: Union[
        Airspace, base.BaseGeometry, Tuple[float, float, float, float]
    ],
) -> T:
    """Returns the part of the DataFrame with coordinates located within the
    bounding box of the shape passed in parameter.

        The bounds parameter can be:

        - an Airspace,
        - a shapely Geometry,
        - a tuple of floats (west, south, east, north)

    """

    if isinstance(bounds, Airspace):
        bounds = bounds.flatten().bounds

    if isinstance(bounds, base.BaseGeometry):
        bounds = bounds.bounds

    west, south, east, north = bounds

    query = "{0} <= longitude <= {2} and {1} <= latitude <= {3}"
    query = query.format(*bounds)

    return geography.query(query)


def _flight_intersects(
    flight: Flight, shape: Union[Airspace, base.BaseGeometry]
) -> bool:
    """Returns True if the trajectory is inside the given shape.

        - If an Airspace is passed, the 3D trajectory is compared to each layers
          constituting the airspace, with corresponding altitude limits.
        - If a shapely Geometry is passed, the 2D trajectory alone is
        considered.

    """
    linestring = flight.airborne().linestring
    if linestring is None:
        return False
    if isinstance(shape, base.BaseGeometry):
        return not linestring.intersection(shape).is_empty
    for layer in shape:
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

setattr(Flight, "inside_bbox", inside_bbox)
setattr(Traffic, "inside_bbox", lazy_evaluation(default=True)(inside_bbox))

setattr(Flight, "intersects", _flight_intersects)
