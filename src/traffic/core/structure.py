from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import rich.repr

from shapely.geometry import GeometryCollection, LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..data.basic.runways import RunwayAirport
from .mixins import FormatMixin, HBoxMixin, PointMixin, ShapelyMixin

if TYPE_CHECKING:
    import altair as alt
    from cartes.osm import Overpass
    from ipyleaflet import GeoData as LeafletGeoData
    from ipyleaflet import Map
    from ipyleaflet import Polyline as LeafletPolyline
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes


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
        self,
        ax: "Axes",
        text_kw: Optional[Mapping[str, Any]] = None,
        shift: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> List["Artist"]:  # coverage: ignore
        from ..visualize.markers import atc_tower

        return super().plot(
            ax, text_kw, shift, **{**{"marker": atc_tower, "s": 400}, **kwargs}
        )


@rich.repr.auto()
class Airport(
    FormatMixin, HBoxMixin, AirportNamedTuple, PointMixin, ShapelyMixin
):
    def __rich_repr__(self) -> rich.repr.Result:
        yield "icao", self.icao
        if self.iata:
            yield "iata", self.iata
        if self.name:
            yield "name", self.name
        if self.country:
            yield "country", self.country
        if self.latitude and self.longitude:
            yield "latitude", self.latitude
            yield "longitude", self.longitude
        if self.altitude:
            yield "altitude", self.altitude

    def _repr_html_(self) -> str:
        title = "<h4><b>Airport</b></h4>"
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
        title += f"<code>{self.icao}/{self.iata}</code><br/>"
        title += f"    {self.latlon}, altitude: {self.altitude}"
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __getattr__(self, name: str) -> "Overpass":
        if name not in self._openstreetmap().data.aeroway.unique():
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
        else:
            res = self._openstreetmap().query(f"aeroway == '{name}'")
            return res.drop(
                columns=list(
                    x for x in res.data.columns if x.startswith("name:")
                )
            )

    @lru_cache()
    def _openstreetmap(self) -> "Overpass":  # coverage: ignore
        from cartes.osm import Overpass

        return Overpass.request(
            area={"icao": self.icao, "as_": "airport"},
            nwr=[dict(aeroway=True, area="airport")],
        )

    def leaflet(self, **kwargs: Any) -> "LeafletGeoData":
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

    def map_leaflet(self, **kwargs: Any) -> "Map":
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

    @property
    def __geo_interface__(self) -> Optional[Dict[str, Any]]:
        df = self._openstreetmap().query('type_ != "node"')
        if df.data.shape[0] == 0:
            return None
        return df.__geo_interface__  # type: ignore

    @property
    def shape(self) -> BaseGeometry:
        osm = self._openstreetmap()
        if "aeroway" not in osm.data.columns:
            return GeometryCollection()
        return unary_union(osm.query('type_ != "node"').data.geometry)

    @property
    def point(self) -> AirportPoint:
        p = AirportPoint()
        p.latitude, p.longitude = self.latlon
        p.name = self.icao
        return p

    @property
    def runways(self) -> Optional[RunwayAirport]:
        """
        Get runway information associated with the airport.

        >>> airports['EHAM'].runways
          latitude   longitude   bearing   name
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          52.3       4.783       41.36     04
          52.31      4.803       221.4     22
          52.29      4.734       57.93     06
          52.3       4.778       238       24
          52.32      4.746       86.65     09
          52.32      4.797       266.7     27
          52.33      4.74        183       18C
          52.3       4.737       2.997     36C
          52.32      4.78        183       18L
          52.29      4.777       3.002     36R
        ... (2 more entries)"""
        from ..data import runways

        return runways[self]

    def geoencode(
        self,
        footprint: Union[bool, Dict[str, Dict[str, Any]]] = True,
        runways: Union[bool, Dict[str, Dict[str, Any]]] = True,
        labels: Union[bool, Dict[str, Dict[str, Any]]] = True,
        **kwargs: Any,
    ) -> "alt.LayerChart":  # coverage: ignore
        import altair as alt

        base = alt.Chart(self).mark_geoshape()  # type: ignore
        cumul = []
        if footprint:
            params: Dict[str, Dict[str, Any]] = dict(
                aerodrome=dict(color="gainsboro", opacity=0.5),
                apron=dict(color="darkgray", opacity=0.5),
                terminal=dict(color="#888888"),
                hangar=dict(color="#888888"),
                taxiway=dict(filled=False, color="silver", strokeWidth=1.5),
            )
            if isinstance(footprint, dict):
                params = {**params, **footprint}

            for key, value in params.items():
                cumul.append(
                    base.transform_filter(
                        f"datum.aeroway == '{key}'"
                    ).mark_geoshape(**value)
                )
        if runways and self.runways is not None:
            if isinstance(runways, dict):
                cumul.append(self.runways.geoencode(mode="geometry", **runways))
            else:
                cumul.append(self.runways.geoencode(mode="geometry"))
        if labels and self.runways is not None:
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
        footprint: Union[bool, Optional[Dict[str, Any]]] = True,
        runways: Union[bool, Optional[Dict[str, Any]]] = False,
        labels: Union[bool, Optional[Dict[str, Any]]] = False,
        **kwargs,
    ):  # coverage: ignore
        default_footprint = dict(
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
            footprint = {**default_footprint, **footprint}

        if footprint is True:
            footprint = default_footprint
            # runways can come from OSM or from the runway database
            # since options may clash, this should fix it
            if isinstance(runways, dict):
                footprint["runway"] = runways

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

    def __getstate__(self) -> Any:
        return self.__dict__

    def __setstate__(self, d: Any) -> Any:
        self.__dict__.update(d)


@rich.repr.auto()
class Navaid(NavaidTuple, PointMixin):
    def __getattr__(self, name: str) -> float:
        if name == "lat":
            return self.latitude
        if name == "lon":
            return self.longitude
        if name == "alt":
            return self.altitude
        raise AttributeError()

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.name
        yield "type", self.type
        yield "latitude", self.latitude
        yield "longitude", self.longitude
        if self.type in {"DME", "NDB", "TACAN", "VOR"}:
            yield "altitude", self.altitude
            if self.description is not None:
                yield "description", self.description
            yield (
                "frequency",
                f"{self.frequency}{'kHz' if self.type=='NDB' else 'MHz'}",
            )

    def __repr__(self) -> str:
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


@rich.repr.auto()
class Route(HBoxMixin, ShapelyMixin):
    def __init__(self, name: str, navaids: List[Navaid]):
        self.name = name
        self.navaids = navaids

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.name
        yield "navaids", [navaid.name for navaid in self.navaids]

    @property
    def shape(self) -> LineString:
        return LineString(list((x.longitude, x.latitude) for x in self.navaids))

    def leaflet(self, **kwargs: Any) -> "Optional[LeafletPolyline]":
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

    def _info_html(self) -> str:
        title = f"<h4><b>Route {self.name}</b></h4>"
        # title += f"flies through {', '.join(self.navaids)}.<br/>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __getitem__(self, elts: Tuple[str, str]) -> "Route":
        elt1, elt2 = elts
        names = list(navaid.name for navaid in self.navaids)
        idx1, idx2 = names.index(elt1), names.index(elt2)
        if idx1 == idx2:
            raise RuntimeError("The two references must be different")
        # fmt: off
        if idx1 > idx2:
            return Route(
                name=self.name + f" between {elt1} and {elt2}",
                navaids=self.navaids[idx2:idx1+1][::-1],
            )
        else:
            return Route(
                name=self.name + f" between {elt1} and {elt2}",
                navaids=self.navaids[idx1:idx2 + 1],
            )
        # fmt: on

    def plot(self, ax: "Axes", **kwargs: Any) -> None:  # coverage: ignore
        if "color" not in kwargs:
            kwargs["color"] = "#aaaaaa"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5
        if "linewidth" not in kwargs and "lw" not in kwargs:
            kwargs["linewidth"] = 0.8
        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["linestyle"] = "dashed"
        if "projection" in ax.__dict__:
            from cartopy.crs import PlateCarree

            kwargs["transform"] = PlateCarree()

        ax.plot(*self.shape.xy, **kwargs)


def patch_leaflet() -> None:
    from ..visualize.leaflet import (
        airport_leaflet,
        airport_map_leaflet,
        route_leaflet,
    )

    Airport.leaflet = airport_leaflet  # type: ignore
    Airport.map_leaflet = airport_map_leaflet  # type: ignore
    Route.leaflet = route_leaflet  # type: ignore


try:
    patch_leaflet()
except Exception:
    pass
