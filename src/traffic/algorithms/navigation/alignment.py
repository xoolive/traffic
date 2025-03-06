import logging
from typing import Iterable, Iterator, Sequence

import numpy as np

from ...core import Flight, FlightPlan
from ...core.mixins import PointMixin

_log = logging.getLogger(__name__)


class BeaconTrackBearingAlignment:
    """Iterates on segments of trajectories aligned with one of the given
    navigational beacons passed in parameter.

    The name of the navigational beacon is assigned in a new column
    `navaid`.

    :param points: a set of point or identifiers constituting the flight plan.
      A :class:`~traffic.core.FlightPlan` structure can also be provided.

    :param angle_precision: in degrees. The difference between the track angle
      and the bearing with respect to the point must be less than this
      threshold.

    :param time_precision: The maximum time interval during which the difference
      between the angles can exceed the threshold.

    :param min_time: Only segments with a duration of at least ``min_time``
      will be yielded.

    :param min_distance: The minimal distance to a given point must be above the
      ``min_distance`` value to be returned.

    Usage:

    >>> from traffic.data import navaids
    >>> from traffic.data.samples import elal747
    >>> subset = elal747.skip("2h30min").first("2h30min")
    >>> for segment in subset.aligned([navaids['KAVOS'], navaids['PEDER']]):
    ...     print(f"aligned on {segment.navaid_max} for {segment.duration}")
    aligned on KAVOS for 0 days 00:07:00
    aligned on PEDER for 0 days 00:05:40

    See also: :ref:`How to infer a flight plan from a trajectory?`

    """

    points: Sequence[PointMixin]

    def __init__(
        self,
        points: str | PointMixin | Iterable[PointMixin] | FlightPlan,
        angle_precision: int = 1,
        time_precision: str = "2 min",
        min_time: str = "30s",
        min_distance: int = 80,
    ) -> None:
        self.angle_precision = angle_precision
        self.time_precision = time_precision
        self.min_time = min_time
        self.min_distance = min_distance

        if isinstance(points, str):
            from ...data import navaids

            navaid = navaids[points]
            if navaid is None:
                _log.warning(f"Navaid {points} unknown")
                self.points = []
            else:
                self.points = [navaid]
        elif isinstance(points, PointMixin):
            self.points = [points]
        elif isinstance(points, FlightPlan):
            self.points = points.all_points
        else:
            self.points = list(points)

    def apply(self, flight: Flight) -> Iterator[Flight]:
        for navpoint in self.points:
            tentative = (
                flight.distance(navpoint)
                .bearing(navpoint)
                .assign(
                    shift=lambda df: df.distance
                    * (np.radians(df.bearing - df.track).abs()),
                    delta=lambda df: (df.bearing - df.track).abs(),
                )
                .query(f"delta < {self.angle_precision} and distance < 500")
            )
            if tentative is not None:
                for segment in tentative.split(self.time_precision):
                    if (
                        segment.longer_than(self.min_time)
                        and segment.min("distance") < self.min_distance
                    ):
                        yield segment.assign(navaid=navpoint.name)
