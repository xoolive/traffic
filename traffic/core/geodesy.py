"""
This module contains a set of geodesy functions wrapped around all what is
available in the `pyproj <https://pyproj4.github.io/pyproj/stable/>`_ library.
All angles are in degrees, all distances are in meter.

"""
from __future__ import annotations

from typing import Any, Iterable, List, TypeVar

from pyproj import Geod
from shapely.geometry import LineString, Point, base

# It is necessary to keep the list[float] because we need indexation later
F = TypeVar("F", float, Iterable[float], List[float])


def distance(
    lat1: F, lon1: F, lat2: F, lon2: F, *args: Any, **kwargs: Any
) -> F:
    """Computes the distance(s) between two points (or arrays of points).

    :param lat1: latitude value(s)
    :param lon1: longitude value(s)
    :param lat2: latitude value(s)
    :param lon2: longitude value(s)

    """
    geod = Geod(ellps="WGS84")
    angle1, angle2, dist1 = geod.inv(lon1, lat1, lon2, lat2, *args, **kwargs)
    return dist1  # type: ignore


def bearing(lat1: F, lon1: F, lat2: F, lon2: F, *args: Any, **kwargs: Any) -> F:
    """Computes the distance(s) between two points (or arrays of points).

    :param lat1: latitude value(s)
    :param lon1: longitude value(s)
    :param lat2: latitude value(s)
    :param lon2: longitude value(s)

    :return: the bearing angle, in degrees, from the first point to the second

    """
    geod = Geod(ellps="WGS84")
    angle1, angle2, dist1 = geod.inv(lon1, lat1, lon2, lat2, *args, **kwargs)
    return angle1  # type: ignore


def destination(
    lat: F, lon: F, bearing: F, distance: F, *args: Any, **kwargs: Any
) -> tuple[F, F, F]:
    """Computes the point you reach from a set of coordinates, moving in a
    given direction for a given distance.

    :param lat: latitude value(s)
    :param lon: longitude value(s)
    :param bearing: bearing value(s)
    :param distance: distance value(s)

    :return: a tuple with latitude value(s), longitude value(s) and bearing
        from the destination point back to the origin, all in degrees.

    """
    geod = Geod(ellps="WGS84")
    lon_, lat_, back_ = geod.fwd(lon, lat, bearing, distance, *args, **kwargs)
    return lat_, lon_, back_


def greatcircle(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *args: Any,
    **kwargs: Any,
) -> list[tuple[float, float]]:
    """Computes a list of points making the great circle between two points.

    :param lat1: latitude value
    :param lon1: longitude value
    :param lat2: latitude value
    :param lon2: longitude value

    :return: a tuple with latitude values, longitude values, all in degrees.

    """
    geod = Geod(ellps="WGS84")
    return [
        (lat, lon)
        for (lon, lat) in geod.npts(lon1, lat1, lon2, lat2, *args, **kwargs)
    ]


def mrr_diagonal(geom: base.BaseGeometry) -> float:
    """
    Returns the length of the diagonal of the minimum rotated rectangle around
    a given shape.

    Consider using a :meth:`~traffic.core.mixins.ShapelyMixin.project_shape`
    method before applying this method if you need a distance in meters.

    """
    if len(geom) <= 1:
        return 0
    if len(geom) == 2:
        return distance(  # type: ignore
            lat1=geom[0].y, lon1=geom[0].x, lat2=geom[1].y, lon2=geom[1].x
        )
    mrr = LineString(geom).minimum_rotated_rectangle
    if isinstance(mrr, Point):
        return 0
    try:  # in most cases, mrr is a Polygon
        x, y = mrr.exterior.coords.xy
    except AttributeError:  # then it should be a LineString
        p0, p1 = mrr.coords[0], mrr.coords[-1]
        return distance(p0[1], p0[0], p1[1], p1[0])  # type: ignore
    return distance(y[0], x[0], y[2], x[2])  # type: ignore
