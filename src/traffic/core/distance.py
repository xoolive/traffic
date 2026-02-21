import logging
from typing import TYPE_CHECKING

import pitot.geodesy as geo

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .mixins import PointLike

_log = logging.getLogger(__name__)


def closest_point(
    data: pd.DataFrame,
    point: None | "PointLike" = None,
    *,
    latitude: None | float = None,
    longitude: None | float = None,
) -> pd.Series:
    """Return the row in ``data`` closest to a reference point.

    A reference can be provided either as a ``point`` object exposing
    ``latitude``, ``longitude`` and ``name`` attributes, or via explicit
    ``latitude`` and ``longitude`` values.

    :param data: a Dataframe containing ``latitude`` and ``longitude`` columns
    :param point: optional point-like object used as reference
    :param latitude: reference latitude when ``point`` is not provided
    :param longitude: reference longitude when ``point`` is not provided
    :return: closest row as a series, enriched with ``distance`` and ``point``
    """
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
        (latitude * np.ones(len(data.latitude))).astype(np.float64),
        (longitude * np.ones(len(data.longitude))).astype(np.float64),
    )
    argmin = dist_vect.argmin()
    elt = data.iloc[argmin]
    return pd.Series(
        {**dict(elt), **{"distance": dist_vect[argmin], "point": name}},
        name=elt.name,
    )


def minimal_angular_difference(angle1: float, angle2: float) -> float:
    """Return the minimal difference between two angles.

    The result is always in the range [0, 180].

    :param angle1: first angle in degrees
    :param angle2: second angle in degrees
    :return: minimal angular difference in degrees
    """
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)
