import logging
from typing import TYPE_CHECKING, Optional

import pitot.geodesy as geo

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .mixins import PointMixin

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
    """
    Calculate the min diff between two angles, considering circularity.

    Parameters:
        angle1 (float): First angle in degrees.
        angle2 (float): Second angle in degrees.

    Returns:
        float: Minimal angular difference in degrees.
    """
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)
