from typing import Any, Iterable, List, Tuple, TypeVar

from pyproj import Geod
from shapely.geometry import LineString, Point, base

T = TypeVar("T", float, List[float], Iterable[float])


def distance(
    lat1: T, lon1: T, lat2: T, lon2: T, *args: Any, **kwargs: Any
) -> T:
    geod = Geod(ellps="WGS84")
    angle1, angle2, dist1 = geod.inv(lon1, lat1, lon2, lat2, *args, **kwargs)
    return dist1  # type: ignore


def bearing(lat1: T, lon1: T, lat2: T, lon2: T, *args: Any, **kwargs: Any) -> T:
    geod = Geod(ellps="WGS84")
    angle1, angle2, dist1 = geod.inv(lon1, lat1, lon2, lat2, *args, **kwargs)
    return angle1  # type: ignore


def destination(
    lat: T, lon: T, bearing: T, distance: T, *args: Any, **kwargs: Any
) -> Tuple[T, T, T]:
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
) -> List[Tuple[float, float]]:
    geod = Geod(ellps="WGS84")
    return [
        (lat, lon)
        for (lon, lat) in geod.npts(lon1, lat1, lon2, lat2, *args, **kwargs)
    ]


def mrr_diagonal(geom: base.BaseGeometry) -> float:
    """
    Returns the length of the diagonal of the minimum rotated rectangle.
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
