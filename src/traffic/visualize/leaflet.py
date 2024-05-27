from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from ipyleaflet import GeoData, Map, Marker, MarkerCluster, Polygon, Polyline
from ipywidgets import HTML

from shapely.geometry import LineString

if TYPE_CHECKING:
    from ..core import Flight
    from ..core.structure import Airport

_old_add = Map.add


def flight_leaflet(self: Any, **kwargs: Any) -> Optional[Polyline]:
    """Returns a Leaflet layer to be directly added to a Map.

    The elements passed as kwargs as passed as is to the PolyLine
    constructor.

    Example usage:

    >>> from ipyleaflet import Map
    >>> # Center the map near the landing airport
    >>> m = Map(center=flight.at().latlon, zoom=7)
    >>> m.add(flight)
    >>> m.add(flight.leaflet(color='red'))
    >>> m

    """
    shape = self.shape
    if shape is None:
        return None

    kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
    return Polyline(
        locations=list((lat, lon) for (lon, lat, *_) in shape.coords),
        **kwargs,
    )


def flight_map_leaflet(
    self: Any,
    *,
    zoom: int = 7,
    highlight: Optional[
        Dict[
            str,
            Union[str, "Flight", Callable[["Flight"], Optional["Flight"]]],
        ]
    ] = None,
    airport: Union[None, str, "Airport"] = None,
    **kwargs: Any,
) -> Optional[Map]:
    from ..core import Flight
    from ..data import airports

    last_position = self.query("latitude == latitude").at()
    if last_position is None:
        return None

    _airport = airports[airport] if isinstance(airport, str) else airport

    if "center" not in kwargs:
        if _airport is not None:
            kwargs["center"] = _airport.latlon
        else:
            kwargs["center"] = (
                self.data.latitude.mean(),
                self.data.longitude.mean(),
            )

    m = Map(zoom=zoom, **kwargs)

    if _airport is not None:
        m.add(_airport)

    elt = m.add(self.leaflet())
    elt.popup = HTML()
    elt.popup.value = self._info_html()

    if highlight is None:
        highlight = dict()

    for color, value in highlight.items():
        if isinstance(value, str):
            value = getattr(Flight, value, None)  # type: ignore
            if value is None:
                continue
        assert not isinstance(value, str)
        f: Optional[Flight]
        if isinstance(value, Flight):
            f = value
        else:
            f = value(self)
        if f is not None:
            m.add(f, color=color)

    return m


def traffic_map_leaflet(
    self: Any,
    *,
    zoom: int = 7,
    highlight: Optional[
        Dict[
            str, Union[str, "Flight", Callable[["Flight"], Optional["Flight"]]]
        ]
    ] = None,
    airport: Union[None, str, "Airport"] = None,
    **kwargs: Any,
) -> Optional[Map]:
    from ..core import Flight
    from ..data import airports

    _airport = airports[airport] if isinstance(airport, str) else airport

    if "center" not in kwargs:
        if _airport is not None:
            kwargs["center"] = _airport.latlon
        else:
            kwargs["center"] = (
                self.data.latitude.mean(),
                self.data.longitude.mean(),
            )

    m = Map(zoom=zoom, **kwargs)

    if _airport is not None:
        m.add(_airport)

    for flight in self:
        if flight.query("latitude == latitude"):
            elt = m.add(flight.leaflet())
            elt.popup = HTML()
            elt.popup.value = flight._info_html()

        if highlight is None:
            highlight = dict()

        for color, value in highlight.items():
            if isinstance(value, str):
                value = getattr(Flight, value, None)  # type: ignore
                if value is None:
                    continue
            assert not isinstance(value, str)
            f: Optional[Flight]
            if isinstance(value, Flight):
                f = value
            else:
                f = value(flight)
            if f is not None:
                m.add(f, color=color)

    return m


def point_leaflet(self: Any, **kwargs: Any) -> Marker:
    """Returns a Leaflet layer to be directly added to a Map.

    The elements passed as kwargs as passed as is to the Marker constructor.
    """

    default = dict()
    if hasattr(self, "name"):
        default["title"] = str(self.name)

    kwargs = {**default, **kwargs}
    marker = Marker(location=(self.latitude, self.longitude), **kwargs)

    label = HTML()
    label.value = repr(self)
    marker.popup = label

    return marker


def statevector_leaflet(self: Any, **kwargs: Any) -> MarkerCluster:
    """Returns a Leaflet layer to be directly added to a Map.

    The elements passed as kwargs as passed as is to the Marker constructor.
    """
    point_list = list(p.leaflet(title=p.callsign, **kwargs) for p in self)
    return MarkerCluster(markers=point_list)


def flightplan_leaflet(self: Any, **kwargs: Any) -> Optional[Polyline]:
    """Returns a Leaflet layer to be directly added to a Map.

    The elements passed as kwargs as passed as is to the PolyLine
    constructor.
    """

    shape = self.shape
    if shape is None:
        return None

    coords: Union[List[List[Tuple[float, float]]], List[Tuple[float, float]]]
    if isinstance(shape, LineString):
        coords = list((lat, lon) for (lon, lat, *_) in shape.coords)
    else:
        # In case a FlightPlan could not resolve all parts
        coords = list(
            list((lat, lon) for (lon, lat, *_) in s.coords) for s in shape.geoms
        )

    kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
    return Polyline(locations=coords, **kwargs)


def airspace_leaflet(self: Any, **kwargs: Any) -> Polygon:
    """Returns a Leaflet layer to be directly added to a Map.

    The elements passed as kwargs as passed as is to the Polygon
    constructor.
    """

    shape = self.flatten()

    kwargs = {**dict(weight=3), **kwargs}
    coords: List[Any] = []

    def unfold(shape: Any) -> Iterator[Any]:  # actually Polygon
        yield shape.exterior
        yield from shape.interiors

    if shape.geom_type == "Polygon":
        coords = list(
            list((lat, lon) for (lon, lat, *_) in x.coords)
            for x in unfold(shape)
        )
    else:
        coords = list(
            list((lat, lon) for (lon, lat, *_) in x.coords)
            for piece in shape.geoms
            for x in unfold(piece)
        )

    return Polygon(locations=coords, **kwargs)


def airspace_map_leaflet(
    self: Any,
    *,
    zoom: int = 6,
    **kwargs: Any,
) -> Map:
    if "center" not in kwargs:
        (lon, lat), *_ = self.shape.centroid.coords
        kwargs["center"] = (lat, lon)

    m = Map(zoom=zoom, **kwargs)

    elt = m.add(self)
    elt.popup = HTML()
    elt.popup.value = self.designator

    return m


def airport_leaflet(self: Any, **kwargs: Any) -> GeoData:
    return GeoData(
        geo_dataframe=self._openstreetmap().query('aeroway == "runway"').data,
        style={**{"color": "#79706e", "weight": 6}, **kwargs},
    )


def airport_map_leaflet(self: Any, **kwargs: Any) -> Map:
    m = Map(center=self.latlon, zoom=13)
    m.add(self.leaflet(**kwargs))
    return m


def route_leaflet(self: Any, **kwargs: Any) -> Optional[Polyline]:
    """Returns a Leaflet layer to be directly added to a Map.

    The elements passed as kwargs as passed as is to the PolyLine
    constructor.
    """

    if self.shape is None:
        return None

    coords = list((lat, lon) for (lon, lat, *_) in self.shape.coords)
    kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
    return Polyline(locations=coords, **kwargs)


def map_add(
    _map: Map,
    elt: Any,
    with_points: bool = True,
    **kwargs: Any,
) -> Any:
    from ..core import (
        Airspace,
        Flight,
        FlightIterator,
        FlightPlan,
        StateVectors,
    )
    from ..core.mixins import PointMixin
    from ..core.structure import Airport, Route

    if any(
        isinstance(elt, c)
        for c in (
            Airport,
            Airspace,
            Flight,
            FlightPlan,
            PointMixin,
            StateVectors,
        )
    ):
        layer = elt.leaflet(**kwargs)
        _old_add(_map, layer)
        return layer
    if isinstance(elt, Route):
        layer = elt.leaflet(**kwargs)
        _old_add(_map, layer)
        if with_points:
            for point in elt.navaids:
                map_add(_map, point, **kwargs)
        return layer
    if isinstance(elt, FlightIterator):
        for segment in elt:
            map_add(_map, segment, **kwargs)
        return
    return _old_add(_map, elt)


def monkey_patch() -> None:
    Map.add = map_add
