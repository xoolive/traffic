from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from cartopy.crs import PlateCarree
from shapely.geometry import LineString, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union

import altair as alt
from cartotools.osm import request, tags

from .. import cache_expiration
from ..drawing import Nominatim
from .mixins import HBoxMixin, PointMixin, ShapelyMixin

request.cache_expiration = cache_expiration


class AirportNamedTuple(NamedTuple):

    altitude: float
    country: str
    iata: str
    icao: str
    latitude: float
    longitude: float
    name: str


class Airport(HBoxMixin, AirportNamedTuple, PointMixin, ShapelyMixin):
    def __repr__(self) -> str:
        short_name = (
            self.name.replace("International", "")
            .replace("Airport", "")
            .strip()
        )
        return f"{self.icao}/{self.iata}: {short_name}"

    def _repr_html_(self) -> str:
        title = f"<b>{self.name.strip()}</b> ({self.country}) "
        title += f"<code>{self.icao}/{self.iata}</code>"
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __getattr__(self, name) -> Dict[str, Any]:
        if not name.startswith("osm_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
        else:
            values = self.osm_request().ways.values()
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(elt[1]),
                        "properties": elt[0]["tags"],
                    }
                    for elt in values
                    if elt[0]["tags"]["aeroway"] == name[4:]
                ],
            }

    def osm_tags(self) -> Set[str]:
        return set(
            elt[0]["tags"]["aeroway"]
            for elt in self.osm_request().ways.values()
        )

    @lru_cache()
    def osm_request(self) -> Nominatim:  # coverage: ignore

        if self.runways is not None:
            lon1, lat1, lon2, lat2 = self.runways.bounds
            return request(
                (lon1 - 0.02, lat1 - 0.02, lon2 + 0.02, lat2 + 0.02),
                **tags.airport,
            )

        else:
            return request(
                (
                    self.longitude - 0.01,
                    self.latitude - 0.01,
                    self.longitude + 0.01,
                    self.latitude + 0.01,
                ),
                **tags.airport,
            )

    @lru_cache()
    def geojson(self):
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "geometry": mapping(shape),
                    "properties": info["tags"],
                    "type": "Feature",
                }
                for info, shape in self.osm_request().ways.values()
                if info["tags"]["aeroway"] != "aerodrome"
            ],
        }

    def geoencode(
        self,
        footprint: bool = True,
        runways: bool = False,
        labels: bool = False,
    ) -> alt.Chart:  # coverage: ignore
        cumul = []
        if footprint:
            cumul.append(super().geoencode())
        if runways:
            cumul.append(self.runways.geoencode())
        if labels:
            cumul.append(self.runways.geoencode("labels"))
        if len(cumul) == 0:
            raise TypeError(
                "At least one of footprint, runways and labels must be True"
            )
        return alt.layer(*cumul)

    @property
    def shape(self):
        # filter out the contour, helps for useful display
        # list(self.osm_request()),
        return self.osm_request().shape
        return cascaded_union(
            list(
                shape
                for dic, shape in self.osm_request().ways.values()
                if dic["tags"]["aeroway"] != "aerodrome"
            )
        )

    @property
    def point(self):
        p = PointMixin()
        p.latitude, p.longitude = self.latlon
        p.name = self.icao
        return p

    @property
    def runways(self):
        from ..data import runways

        return runways[self]

    def plot(  # type: ignore
        self,
        ax,
        footprint: bool = True,
        runways: Union[bool, Optional[Dict]] = False,
        labels: Union[bool, Optional[Dict]] = False,
        **kwargs,
    ):  # coverage: ignore

        if footprint:
            params = {
                "edgecolor": "silver",
                "facecolor": "None",
                "crs": PlateCarree(),
                **kwargs,
            }
            ax.add_geometries(
                # filter out the contour, helps for useful display
                # list(self.osm_request()),
                list(
                    shape
                    for dic, shape in self.osm_request().ways.values()
                    if dic["tags"]["aeroway"] != "aerodrome"
                ),
                **params,
            )

        if self.runways is None:
            return

        if runways is not False or labels is not False:
            self.runways.plot(
                ax,
                labels=labels is not False,
                text_kw=labels if isinstance(labels, dict) else {},
                **(runways if isinstance(runways, dict) else {}),
            )


class NavaidTuple(NamedTuple):

    name: str
    type: str
    latitude: float
    longitude: float
    altitude: float
    frequency: Optional[float]
    magnetic_variation: Optional[float]
    description: Optional[str]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class Navaid(NavaidTuple, PointMixin):
    def __getattr__(self, name):
        if name == "lat":
            return self.latitude
        if name == "lon":
            return self.longitude
        if name == "alt":
            return self.altitude

    def __repr__(self):
        if self.type in ["FIX", "DB", "WP"] or self.type != self.type:
            return (
                f"{self.name} ({self.type}): {self.latitude} {self.longitude}"
            )
        else:
            return (
                f"{self.name} ({self.type}): {self.latitude} {self.longitude}"
                f" {self.altitude:.0f} "
                f"{self.description if self.description is not None else ''}"
                f" {self.frequency}{'kHz' if self.type=='NDB' else 'MHz'}"
            )


class Route(HBoxMixin, ShapelyMixin):
    def __init__(self, shape: BaseGeometry, name: str, navaids: List[str]):
        self.shape = shape
        self.name = name
        self.navaids = navaids

    def __repr__(self):
        return f"{self.name} ({', '.join(self.navaids)})"

    def _info_html(self) -> str:
        title = f"<b>Route {self.name}</b><br/>"
        title += f"flies through {', '.join(self.navaids)}.<br/>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __getitem__(self, elts: Tuple[str, str]) -> "Route":
        elt1, elt2 = elts
        idx1, idx2 = self.navaids.index(elt1), self.navaids.index(elt2)
        if idx1 == idx2:
            raise RuntimeError("The two references must be different")
        if idx1 > idx2:
            idx2, idx1 = idx1, idx2
        # fmt: off
        return Route(
            LineString(self.shape.coords[idx1: idx2 + 1]),
            name=self.name + f" between {elt1} and {elt2}",
            navaids=self.navaids[idx1: idx2 + 1],
        )
        # fmt: on

    def plot(self, ax, **kwargs):  # coverage: ignore
        if "color" not in kwargs:
            kwargs["color"] = "#aaaaaa"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5
        if "linewidth" not in kwargs and "lw" not in kwargs:
            kwargs["linewidth"] = 0.8
        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["linestyle"] = "dashed"
        if "projection" in ax.__dict__:
            kwargs["transform"] = PlateCarree()

        ax.plot(*self.shape.xy, **kwargs)
