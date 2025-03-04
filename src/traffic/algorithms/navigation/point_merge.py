import logging
from operator import attrgetter
from typing import Any, Iterator

from typing_extensions import NotRequired, TypedDict

from ...core import Flight
from ...core.mixins import PointMixin
from ...core.structure import Airport

_log = logging.getLogger(__name__)


class PointMergeParams(TypedDict):
    point_merge: str | PointMixin
    secondary_point: NotRequired[None | str | PointMixin]
    distance_interval: NotRequired[tuple[float, float]]
    delta_threshold: NotRequired[float]


class PointMerge:
    """
    Iterates on all point merge segments in a trajectory before landing at a
    given airport.

    Only the ``point_merge`` argument is mandatory but other arguments may
    reduce the number of false positives.

    :param point_merge: The procedure point on which trajectories all align.

    :param secondary_point: In some cases (e.g. Dublin 10R),
        aircraft align to the ``point_merge`` after a segment of almost
        constant distance to a secondary point.

        Most often, the ``secondary_point`` is the ``point_merge`` and can
        be left as ``None``.

    :param distance_interval: A tuple of distances in nautical miles,
        corresponding to lower and upper bound distances in the AIP between
        the constant distance segments and the point merge.

        This parameter is ignored if left as None.

    :param delta_threshold: keep as default

    :param airport: Remove false positives by specifying the landing
        airport. The algorithm will ensure all trajectories are aligned with
        one of the airport's ILS.

    :param runway: Remove false positives by specifying the landing
        runway. The algorithm will ensure all trajectories are aligned with
        the runway's ILS. (ignored if ``airport`` is ``None``)

    Usage:
    See :ref:`How to implement point-merge detection?`

    (new in version 2.8)

    """

    def __init__(
        self,
        point_merge: str | PointMixin | list[PointMergeParams],
        secondary_point: None | str | PointMixin = None,
        distance_interval: None | tuple[float, float] = None,
        delta_threshold: float = 5e-2,
        airport: None | str | Airport = None,
        runway: None | str = None,
        **kwargs: Any,
    ) -> None:
        self.point_merge = point_merge
        self.secondary_point = secondary_point
        self.distance_interval = distance_interval
        self.delta_threshold = delta_threshold
        self.airport = airport
        self.runway = runway
        self.kwargs = kwargs

    def apply(self, flight: Flight) -> Iterator[Flight]:
        if isinstance(self.point_merge, list):
            results = []
            for params in self.point_merge:
                id_ = params.get("secondary_point", params["point_merge"])
                assert id_ is not None
                name = id_ if isinstance(id_, str) else id_.name
                for segment in flight.point_merge(**params):
                    results.append(segment.assign(point_merge=name))
            yield from sorted(results, key=attrgetter("start"))
            return

        from traffic.data import navaids

        navaids_extent = navaids.extent(flight, buffer=1)
        msg = f"No navaid available in the bounding box of Flight {flight}"

        if isinstance(self.point_merge, str):
            if navaids_extent is None:
                _log.warning(msg)
                return None
            point_merge = navaids_extent.get(self.point_merge)
            if point_merge is None:
                _log.warning("Navaid for point_merge not found")
                return None

        if self.secondary_point is None:
            secondary_point = point_merge

        if isinstance(self.secondary_point, str):
            if navaids_extent is None:
                _log.warning(msg)
                return None
            secondary_point = navaids_extent.get(self.secondary_point)
            if secondary_point is None:
                _log.warning("Navaid for secondary_point not found")
                return None

        if self.airport is not None:
            for landing in flight.aligned_on_ils(self.airport, **self.kwargs):
                if self.runway is None or landing.max("ILS") == self.runway:
                    yield from flight.point_merge(
                        point_merge=self.point_merge,
                        secondary_point=self.secondary_point,
                        distance_interval=self.distance_interval,
                        delta_threshold=self.delta_threshold,
                    )
            return

        for segment in flight.aligned_on_navpoint(point_merge):
            before_point = flight.before(segment.start)
            if before_point is None:
                continue
            before_point = before_point.last("10 minutes")
            if before_point is None:
                continue
            lower, upper = (
                self.distance_interval if self.distance_interval else (0, 100)
            )
            constant_distance = (
                before_point.distance(self.secondary_point)
                .diff("distance")
                .query(
                    f"{lower} < distance < {upper} and "
                    f"distance_diff.abs() < {self.delta_threshold}"
                )
            )
            if constant_distance is None:
                continue
            candidate = constant_distance.split("5 seconds").max()
            if candidate is not None and candidate.longer_than("90 seconds"):
                result = flight.between(candidate.start, segment.stop)
                if result is not None:
                    yield result
