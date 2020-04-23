import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
import pandas as pd

from . import geodesy as geo

if TYPE_CHECKING:
    from ..data.basic.airports import Airports  # noqa: F401
    from .mixins import PointMixin  # noqa: F401
    from .structure import Airport


def closest_point(
    data: pd.DataFrame,
    point: Optional["PointMixin"] = None,
    *args,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> pd.Series:

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
    elt = data.iloc[argmin]
    return pd.Series(
        {**dict(elt), **{"distance": dist_vect[argmin], "point": name}},
        name=elt.name,
    )


def guess_airport(
    point: Optional[NamedTuple] = None,
    *args,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    dataset: Optional["Airports"] = None,
    warning_distance: Optional[float] = None,
) -> "Airport":

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

    airport = dataset[airport_data.icao]
    assert airport is not None
    airport.distance = airport_data.distance  # type: ignore

    if warning_distance is not None and airport.distance > warning_distance:
        logging.warning(
            f"Closest airport is more than {warning_distance*1e-3}km away "
            f" (distance={airport.distance})"
        )
    return airport
