# fmt: off

from functools import lru_cache
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes

import altair as alt
from cartes.osm import Overpass
from cartopy.crs import PlateCarree
from shapely.geometry import GeometryCollection, LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..drawing.markers import atc_tower
from .mixins import HBoxMixin, PointMixin, ShapelyMixin

# fmt: on


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
            and self.runway.data.shape[0] > 0
        ):
            title += "<div class='alert alert-warning'><p>"
            title += "<b>Warning!</b> No runway information available in our "
            title += "database. Please consider helping the community by "
            title += "updating the runway information with data provided "
            title += "by OpenStreetMap.</p>"

            url = f"https://ourairports.com/airports/{self.icao}/runways.html"
            title += f"<p>Edit link: <a href='{url}'>{url}</a>.<br/>"
            title += "Check the data in "
            title += f"<code>airports['{self.icao}'].runway</code>"
            title += " and edit the webpage accordingly. You may need to "
            title += " create an account there to be able to edit.</p></div>"

        title += f"<b>{self.name.strip()}</b> ({self.country}) "
        title += f"<code>{self.icao}/{self.iata}</code>"
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __getattr__(self, name: str) -> Overpass:
        if name not in self._openstreetmap().data.aeroway.unique():
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
        else:
            return self._openstreetmap().query(f"aeroway == '{name}'")

    @lru_cache()
    def _openstreetmap(self) -> Overpass:  # coverage: ignore
        return Overpass.request(
            area={"icao": self.icao, "as_": "airport"},
            nwr=[dict(area="airport"), dict(aeroway=True, area="airport")],
        )

    @property
    def __geo_interface__(self):
        return self._openstreetmap().query('type_ != "node"').__geo_interface__

    @property
    def shape(self) -> BaseGeometry:
        osm = self._openstreetmap()
        if "aeroway" not in osm.data.columns:
            return GeometryCollection()
        return unary_union(osm.query('aeroway == "aerodrome"').data.geometry)

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

    def geoencode(  # type: ignore
        self,
        footprint: Union[bool, dict] = True,
        runways: Union[bool, dict] = True,
        labels: Union[bool, dict] = True,
    ) -> alt.Chart:  # coverage: ignore

        base = alt.Chart(self).mark_geoshape()
        cumul = []
        if footprint:
            params = dict(
                aerodrome=dict(color="gainsboro", opacity=0.5),
                apron=dict(color="darkgray", opacity=0.5),
                terminal=dict(color="#888888"),
                hangar=dict(color="#888888"),
                taxiway=dict(filled=False, color="silver", strokeWidth=1.5),
            )
            if isinstance(footprint, dict):
                footprint = {**params, **footprint}
            else:
                footprint = params

            for key, value in footprint.items():
                cumul.append(
                    base.transform_filter(
                        f"datum.aeroway == '{key}'"
                    ).mark_geoshape(**value)
                )
        if runways:
            if isinstance(runways, dict):
                cumul.append(self.runways.geoencode(mode="geometry", **runways))
            else:
                cumul.append(self.runways.geoencode(mode="geometry"))
        if labels:
            if isinstance(labels, dict):
                cumul.append(self.runways.geoencode(mode="labels", **labels))
            else:
                cumul.append(self.runways.geoencode(mode="labels"))
        if len(cumul) == 0:
            raise TypeError(
                "At least one of footprint, runways and labels must be True"
            )
        return alt.layer(*cumul)

    def plot(  # type: ignore
        self,
        ax,
        footprint: Union[bool, Optional[Dict]] = True,
        runways: Union[bool, Optional[Dict]] = False,
        labels: Union[bool, Optional[Dict]] = False,
        **kwargs,
    ):  # coverage: ignore

        if footprint is True:
            footprint = dict(
                by="aeroway",
                aerodrome=dict(color="gainsboro", alpha=0.5),
                apron=dict(color="darkgray", alpha=0.5),
                taxiway=dict(color="darkgray"),
                terminal=dict(color="black"),
                # MUTE the rest
                gate=dict(alpha=0),
                parking_position=dict(alpha=0),
                holding_position=dict(alpha=0),
                tower=dict(alpha=0),
                helipad=dict(alpha=0),
                jet_bridge=dict(alpha=0),
                aerobridge=dict(alpha=0),
                navigationaid=dict(
                    papi=dict(alpha=0), approach_light=dict(alpha=0)
                ),
                windsock=dict(alpha=0),
            )

        if isinstance(footprint, dict):
            self._openstreetmap().plot(ax, **footprint)

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
