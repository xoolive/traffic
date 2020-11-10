# fmt: off

from functools import lru_cache
from itertools import chain
from typing import (
    Any, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Union
)

import altair as alt
from cartopy.crs import PlateCarree
from cartotools.osm import request, tags
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes
from shapely.geometry import LineString, Polygon, mapping, polygon
from shapely.geometry.base import BaseGeometry

from .. import cache_expiration
from ..drawing import Nominatim
from ..drawing.markers import atc_tower
from .mixins import HBoxMixin, PointMixin, ShapelyMixin

# fmt: on

request.cache_expiration = cache_expiration


class AirportNamedTuple(NamedTuple):

    altitude: float
    country: str
    iata: str
    icao: str
    latitude: float
    longitude: float
    name: str


class AirportPoint(PointMixin):
    def plot(
        self, ax: Axes, text_kw=None, shift=None, **kwargs
    ) -> List[Artist]:  # coverage: ignore
        return super().plot(
            ax, text_kw, shift, **{**{"marker": atc_tower, "s": 400}, **kwargs}
        )


class Airport(HBoxMixin, AirportNamedTuple, PointMixin, ShapelyMixin):
    def __repr__(self) -> str:
        short_name = (
            self.name.replace("International", "")
            .replace("Airport", "")
            .strip()
        )
        return f"{self.icao}/{self.iata}: {short_name}"

    def _repr_html_(self) -> str:
        title = ""
        if (
            self.runways is not None
            and self.runways.shape.is_empty
            and len(self.osm_runway["features"]) > 0
        ):
            title += "<div class='alert alert-warning'><p>"
            title += "<b>Warning!</b> No runway information available in our "
            title += "database. Please consider helping the community by "
            title += "updating the runway information with data provided "
            title += "by OpenStreetMap.</p>"

            url = f"https://ourairports.com/airports/{self.icao}/runways.html"
            title += f"<p>Edit link: <a href='{url}'>{url}</a>.<br/>"
            title += "Check the data in "
            title += f"<code>airports['{self.icao}'].osm_runway</code>"
            title += " and edit the webpage accordingly. You may need to "
            title += " create an account there to be able to edit.</p></div>"

        title += f"<b>{self.name.strip()}</b> ({self.country}) "
        title += f"<code>{self.icao}/{self.iata}</code>"
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __getattr__(self, name) -> Dict[str, Any]:
        if not name.startswith("osm_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
        else:
            values = self.osm_values()
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

    def osm_values(self) -> Iterator[Tuple[Dict[str, Any], BaseGeometry]]:
        return chain(
            (
                (dict_, shape_)
                for dict_, shape_ in self.osm_request().nodes.values()
                if "tags" in dict_ and "aeroway" in dict_["tags"]
            ),
            self.osm_request().ways.values(),
        )

    def osm_tags(self) -> Set[str]:
        return set(elt[0]["tags"]["aeroway"] for elt in self.osm_values())

    @lru_cache()
    def osm_request(self) -> Nominatim:  # coverage: ignore

        if self.runways is not None and not self.runways.shape.is_empty:
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
                    "geometry": mapping(
                        polygon.orient(shape, -1)
                        if type(shape) == Polygon
                        else shape
                    ),
                    "properties": info["tags"],
                    "type": "Feature",
                }
                for info, shape in self.osm_request().ways.values()
                if info["tags"]["aeroway"] != "aerodrome"
            ],
        }

    def geoencode(  # type: ignore
        self, footprint: bool = True, runways: bool = True, labels: bool = True,
    ) -> alt.Chart:  # coverage: ignore
        cumul = []
        if footprint:
            cumul.append(super().geoencode(fill=""))
        if runways:
            cumul.append(self.runways.geoencode(mode="geometry"))
        if labels:
            cumul.append(self.runways.geoencode(mode="labels"))
        if len(cumul) == 0:
            raise TypeError(
                "At least one of footprint, runways and labels must be True"
            )
        return alt.layer(*cumul).configure_view(opacity=0)

    @property
    def shape(self):
        # filter out the contour, helps for useful display
        return self.osm_request().shape

    @property
    def point(self):
        p = AirportPoint()
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
        if self.type not in {"DME", "NDB", "TACAN", "VOR"}:
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
