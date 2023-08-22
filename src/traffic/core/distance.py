import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import pitot.geodesy as geo

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..data.basic.airports import Airports
    from .mixins import PointMixin
    from .structure import Airport

_log = logging.getLogger(__name__)


def closest_point(
    data: pd.DataFrame,
    point: Optional["PointMixin"] = None,
    *,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> pd.Series:
    if point is not None:
        latitude = point.latitude
        longitude = point.longitude
        name = point.name
    else:
        name = "unnamed"
    assert latitude is not None and longitude is not None
    dist_vect = geo.distance(
        data.latitude.values,
        data.longitude.values,
        latitude * np.ones(len(data.latitude)),
        longitude * np.ones(len(data.longitude)),
    )
    argmin = dist_vect.argmin()
    elt = data.iloc[argmin]
    return pd.Series(
        {**dict(elt), **{"distance": dist_vect[argmin], "point": name}},
        name=elt.name,
    )


def guess_airport(
    point: Optional[NamedTuple] = None,
    *,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    dataset: Optional["Airports"] = None,
    warning_distance: Optional[float] = None,
) -> "Airport":
    from ..core.structure import Airport

    if dataset is None:
        from ..data import airports

        dataset = airports

    # TODO define a protocol instead of PointMixin
    if point is not None:
        longitude = point.longitude  # type: ignore
        latitude = point.latitude  # type: ignore
    if any((longitude is None, latitude is None)):
        raise RuntimeError("latitude or longitude are None")

    airport_data = closest_point(
        dataset.data, latitude=latitude, longitude=longitude
    )
    airport = Airport(
        airport_data.get("altitude"),
        airport_data.get("country"),
        airport_data.get("iata"),
        airport_data.get("icao"),
        airport_data.get("latitude"),
        airport_data.get("longitude"),
        airport_data.get("name"),
    )
    airport.distance = airport_data.distance  # type: ignore

    if warning_distance is not None and airport.distance > warning_distance:
        _log.warning(
            f"Closest airport is more than {warning_distance*1e-3}km away "
            f" (distance={airport.distance})"
        )
    return airport
