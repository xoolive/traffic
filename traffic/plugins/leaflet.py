from typing import Any, List, Optional

from ipyleaflet import Map, Marker, Polygon, Polyline

from traffic.core import Airspace, Flight
from traffic.core.mixins import PointMixin
from traffic.plugins import PluginProvider


def flight_leaflet(flight: Flight, **kwargs) -> Optional[Polyline]:
    shape = flight.shape
    if shape is None:
        return None

    kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
    return Polyline(
        locations=list((lat, lon) for (lon, lat, _) in shape.coords), **kwargs
    )


def airspace_leaflet(airspace: Airspace, **kwargs) -> Polygon:
    shape = airspace.flatten()

    kwargs = {**dict(weight=3), **kwargs}
    coords: List[Any] = []
    if shape.geom_type == "Polygon":
        coords = list((lat, lon) for (lon, lat) in shape.exterior.coords)
    else:
        coords = list(
            list((lat, lon) for (lon, lat) in piece.exterior.coords)
            for piece in shape
        )

    return Polygon(locations=coords, **kwargs)


def point_leaflet(point: PointMixin, **kwargs) -> Marker:

    default = dict()
    if hasattr(point, "name"):
        default["title"] = point.name

    kwargs = {**default, **kwargs}
    return Marker(location=(point.latitude, point.longitude), **kwargs)


_old_add_layer = Map.add_layer


def map_add_layer(_map, elt, **kwargs):
    if any(isinstance(elt, c) for c in (Flight, Airspace, PointMixin)):
        layer = elt.leaflet(**kwargs)
        _old_add_layer(_map, layer)
        return layer
    return _old_add_layer(_map, elt)


class Leaflet(PluginProvider):
    def load_plugin(self):
        setattr(Flight, "leaflet", flight_leaflet)
        setattr(Airspace, "leaflet", airspace_leaflet)
        setattr(PointMixin, "leaflet", point_leaflet)
        setattr(Map, "add_layer", map_add_layer)
