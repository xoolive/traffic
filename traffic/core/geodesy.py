from typing import Iterable, Tuple, TypeVar

from pyproj import Geod

T = TypeVar("T", float, Iterable[float])


def distance(lat1: T, lon1: T, lat2: T, lon2: T, *args, **kwargs) -> T:
    geod = Geod(ellps="WGS84")
    angle1, angle2, dist1 = geod.inv(lon1, lat1, lon2, lat2, *args, **kwargs)
    return dist1


def bearing(lat1: T, lon1: T, lat2: T, lon2: T, *args, **kwargs) -> T:
    geod = Geod(ellps="WGS84")
    angle1, angle2, dist1 = geod.inv(lon1, lat1, lon2, lat2, *args, **kwargs)
    return angle1


def destination(
    lat: T, lon: T, bearing: T, distance: T, *args, **kwargs
) -> Tuple[T, T, T]:
    geod = Geod(ellps="WGS84")
    lon_, lat_, back_ = geod.fwd(lon, lat, bearing, distance, *args, **kwargs)
    return lat_, lon_, back_


def greatcircle(lat1, lon1, lat2, lon2, *args, **kwargs):
    geod = Geod(ellps="WGS84")
    return geod.npts(lon1, lat1, lon2, lat2, *args, **kwargs)
