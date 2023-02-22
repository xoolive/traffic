from typing import Any, Union

from ipyleaflet import Map

from . import Airspace, Flight, FlightIterator, FlightPlan, StateVectors
from .mixins import PointMixin
from .structure import Airport, Route

_old_add = Map.add

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


def map_add(_map: Map, elt: MapElement, **kwargs: Any) -> Any:
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
        _old_add(_map, layer)
        return layer
    if isinstance(elt, Route):
        layer = elt.leaflet(**kwargs)
        _old_add(_map, layer)
        for point in elt.navaids:
            map_add(_map, point, **kwargs)
        return layer
    if isinstance(elt, FlightIterator):
        for segment in elt:
            map_add(_map, segment, **kwargs)
        return
    return _old_add(_map, elt)


def monkey_patch() -> None:
    setattr(Map, "add", map_add)
    setattr(Map, "add_layer", map_add)
