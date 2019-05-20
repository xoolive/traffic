from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
import pandas as pd

from . import geodesy as geo
from .mixins import PointMixin

if TYPE_CHECKING:
    from ..data.basic.airports import Airport  # noqa: F401


class DistanceAirport(NamedTuple):
    distance: float
    airport: "Airport"


class DistancePointTrajectory(NamedTuple):
    distance: float
    name: str
    point: pd.Series


def closest_point(
    data: pd.DataFrame,
    point: Optional[PointMixin] = None,
    *args,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> DistancePointTrajectory:

    if point is not None:
        latitude = point.latitude
        longitude = point.longitude
        name = point.name
    else:
        name = "unnamed"
    dist_vect = geo.distance(
        data.latitude.values,
        data.longitude.values,
        latitude * np.ones(len(data.latitude)),
        longitude * np.ones(len(data.longitude)),
    )
    argmin = dist_vect.argmin()
    return DistancePointTrajectory(dist_vect[argmin], name, data.iloc[argmin])


def guess_airport(
    point: Optional[NamedTuple] = None,
    *args,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> DistanceAirport:
    from ..data import airports

    # TODO define a protocol instead of PointMixin
    if point is not None:
        longitude = point.longitude  # type: ignore
        latitude = point.latitude  # type: ignore
    if any((longitude is None, latitude is None)):
        raise RuntimeError("latitude or longitude are None")

    distance, _, airport = closest_point(
        airports.data, latitude=latitude, longitude=longitude
    )

    airport_handle = airports[airport.icao]
    assert airport_handle is not None
    return DistanceAirport(distance, airport_handle)
