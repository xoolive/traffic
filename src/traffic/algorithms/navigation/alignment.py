import logging
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

from ...core import Flight, FlightPlan
from ...core.mixins import PointMixin

_log = logging.getLogger(__name__)


class AlignmentBase(Protocol):
    def apply(self, flight: Flight) -> Iterator[Flight]: ...


class BeaconTrackBearingAlignment:
    """Iterates on segments of trajectories aligned with one of the given
    navigational beacons passed in parameter.

    The name of the navigational beacon is assigned in a new column
    `navaid`.

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
