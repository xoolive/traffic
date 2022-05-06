from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from ipyleaflet import GeoData, Map, Marker, MarkerCluster, Polygon, Polyline
from ipywidgets import HTML

from shapely.geometry import LineString

from ..core import (
    Airspace,
    Flight,
    FlightIterator,
    FlightPlan,
    StateVectors,
    Traffic,
)
from ..core.mixins import PointMixin
from ..core.structure import Airport, Route


def traffic_map_leaflet(
    traffic: "Traffic",
    *,
    zoom: int = 7,
    highlight: Optional[
        Dict[str, Union[str, Flight, Callable[[Flight], Optional[Flight]]]]
    ] = None,
    airport: Union[None, str, Airport] = None,
    **kwargs: Any,
) -> Optional[Map]:

    from traffic.data import airports

    _airport = airports[airport] if isinstance(airport, str) else airport

    if "center" not in kwargs:
        if _airport is not None:
            kwargs["center"] = _airport.latlon
        else:
            kwargs["center"] = (
                traffic.data.latitude.mean(),
                traffic.data.longitude.mean(),
            )

    m = Map(zoom=zoom, **kwargs)

    if _airport is not None:
        m.add_layer(_airport)

    for flight in traffic:
        if flight.query("latitude == latitude"):
            elt = m.add_layer(flight)
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
                m.add_layer(f, color=color)

    return m


def flight_map_leaflet(
    flight: "Flight",
    *,
    zoom: int = 7,
    highlight: Optional[
        Dict[
            str,
            Union[str, Flight, Callable[[Flight], Optional[Flight]]],
        ]
    ] = None,
    airport: Union[None, str, Airport] = None,
    **kwargs: Any,
) -> Optional[Map]:
    from traffic.core import Flight
    from traffic.data import airports

    last_position = flight.query("latitude == latitude").at()  # type: ignore
    if last_position is None:
        return None

    _airport = airports[airport] if isinstance(airport, str) else airport

    if "center" not in kwargs:
        if _airport is not None:
            kwargs["center"] = _airport.latlon
        else:
            kwargs["center"] = (
                flight.data.latitude.mean(),
                flight.data.longitude.mean(),
            )

    m = Map(zoom=zoom, **kwargs)

    if _airport is not None:
        m.add_layer(_airport)

    elt = m.add_layer(flight)
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
            m.add_layer(f, color=color)

    return m


def flight_leaflet(flight: "Flight", **kwargs: Any) -> Optional[Polyline]:
    """Returns a Leaflet layer to be directly added to a Map.

    .. warning::

        This is only available if the :ref:`Leaflet` plugin is activated. (True
        by default)

    The elements passed as kwargs as passed as is to the PolyLine constructor.

    Example usage:

    >>> from ipyleaflet import Map
    >>> # Center the map near the landing airport
    >>> m = Map(center=flight.at().latlon, zoom=7)
    >>> m.add_layer(flight)  # this works as well with default options
    >>> m.add_layer(flight.leaflet(color='red'))
    >>> m
    """
    shape = flight.shape
    if shape is None:
        return None

    kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
    return Polyline(
        locations=list((lat, lon) for (lon, lat, _) in shape.coords), **kwargs
    )


def flightplan_leaflet(
    flightplan: "FlightPlan", **kwargs: Any
) -> Optional[Polyline]:
    """Returns a Leaflet layer to be directly added to a Map.

    .. warning::

        This is only available if the :ref:`Leaflet` plugin is activated. (True
        by default)

    The elements passed as kwargs as passed as is to the PolyLine constructor.
    """

    shape = flightplan.shape
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


def route_leaflet(route: "Route", **kwargs: Any) -> Optional[Polyline]:
    """Returns a Leaflet layer to be directly added to a Map.

    .. warning::

        This is only available if the :ref:`Leaflet` plugin is activated. (True
        by default)

    The elements passed as kwargs as passed as is to the PolyLine constructor.
    """

    if route.shape is None:
        return None

    coords = list((lat, lon) for (lon, lat, *_) in route.shape.coords)
    kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
    return Polyline(locations=coords, **kwargs)


def airspace_leaflet(airspace: "Airspace", **kwargs: Any) -> Polygon:
    """Returns a Leaflet layer to be directly added to a Map.

    .. warning::

        This is only available if the :ref:`Leaflet` plugin is activated. (True
        by default)

    The elements passed as kwargs as passed as is to the Polygon constructor.
    """
    shape = airspace.flatten()

    kwargs = {**dict(weight=3), **kwargs}
    coords: List[Any] = []

    def unfold(shape: Polygon) -> Iterator[Any]:
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
    airspace: "Airspace",
    *,
    zoom: int = 6,
    **kwargs: Any,
) -> Optional[Map]:

    if "center" not in kwargs:
        (lon, lat), *_ = airspace.shape.centroid.coords
        kwargs["center"] = (lat, lon)

    m = Map(zoom=zoom, **kwargs)

    elt = m.add_layer(airspace)
    elt.popup = HTML()
    elt.popup.value = airspace.designator

    return m


def airport_leaflet(airport: "Airport", **kwargs: Any) -> GeoData:
    return GeoData(
        geo_dataframe=airport._openstreetmap()
        .query('aeroway == "runway"')
        .data,
        style={**{"color": "#79706e", "weight": 6}, **kwargs},
    )


def sv_leaflet(sv: "StateVectors", **kwargs: Any) -> MarkerCluster:
    """Returns a Leaflet layer to be directly added to a Map.

    .. warning::

        This is only available if the :ref:`Leaflet` plugin is activated. (True
        by default)

    The elements passed as kwargs as passed as is to the Marker constructor.
    """
    point_list = list(point_leaflet(p, title=p.callsign, **kwargs) for p in sv)
    return MarkerCluster(markers=point_list)


def point_leaflet(point: "PointMixin", **kwargs: Any) -> Marker:
    """Returns a Leaflet layer to be directly added to a Map.

    .. warning::

        This is only available if the :ref:`Leaflet` plugin  is activated. (True
        by default)

    The elements passed as kwargs as passed as is to the Marker constructor.
    """

    default = dict()
    if hasattr(point, "name"):
        default["title"] = str(point.name)

    kwargs = {**default, **kwargs}
    marker = Marker(location=(point.latitude, point.longitude), **kwargs)

    label = HTML()
    label.value = repr(point)
    marker.popup = label

    return marker


_old_add_layer = Map.add_layer

MapElement = Union[
    Airport,
    Airspace,
    Flight,
    FlightIterator,
    FlightPlan,
    PointMixin,
    Route,
    StateVectors,
]


def map_add_layer(_map: Map, elt: MapElement, **kwargs: Any) -> Any:
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
        layer = elt.leaflet(**kwargs)  # type: ignore
        _old_add_layer(_map, layer)
        return layer
    if isinstance(elt, Route):
        layer = elt.leaflet(**kwargs)  # type: ignore
        _old_add_layer(_map, layer)
        for point in elt.navaids:
            map_add_layer(_map, point, **kwargs)
        return layer
    if isinstance(elt, FlightIterator):
        for segment in elt:
            map_add_layer(_map, segment, **kwargs)
        return
    return _old_add_layer(_map, elt)


def _onload() -> None:
    setattr(Airport, "leaflet", airport_leaflet)
    setattr(Airspace, "leaflet", airspace_leaflet)
    setattr(Flight, "leaflet", flight_leaflet)
    setattr(FlightPlan, "leaflet", flightplan_leaflet)
    setattr(PointMixin, "leaflet", point_leaflet)
    setattr(Route, "leaflet", route_leaflet)
    setattr(StateVectors, "leaflet", sv_leaflet)

    setattr(Airspace, "map_leaflet", airspace_map_leaflet)
    setattr(Flight, "map_leaflet", flight_map_leaflet)
    # setattr(FlightPlan, "map_leaflet", flightplan_map_leaflet)
    setattr(Traffic, "map_leaflet", traffic_map_leaflet)

    setattr(Map, "add_layer", map_add_layer)
