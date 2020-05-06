# fmt: off

from operator import attrgetter
from typing import (
    TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Union, cast
)

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

if TYPE_CHECKING:
    from ..core import Flight  # noqa: 401
    from ..core.mixins import PointMixin  # noqa: 401
    from ..core.structure import Airport, Navaid  # noqa: 401

# fmt: on


class NavigationFeatures:

    # white lies
    shape: Optional[LineString]
    query: Callable[["NavigationFeatures", str], Optional["Flight"]]

    def closest_point(
        self, points: Union[List["PointMixin"], "PointMixin"]
    ) -> pd.Series:
        """Selects the closest point of the trajectory with respect to
        a point or list of points.

        The pd.Series returned by the function is enriched with two fields:
        distance (in meters) and point (containing the name of the closest
        point to the trajectory)

        Example usage:

        .. code:: python

            >>> item = belevingsvlucht.between(
            ...     "2018-05-30 16:00", "2018-05-30 17:00"
            ... ).closest_point(  # type: ignore
            ...     [
            ...         airports["EHLE"],  # type: ignore
            ...         airports["EHAM"],  # type: ignore
            ...         navaids["NARAK"],  # type: ignore
            ...     ]
            ... )
            >>> f"{item.point}, {item.distance:.2f}m"
            "Lelystad Airport, 49.11m"

        """
        from ..core.distance import closest_point as cp

        # The following cast secures the typing
        self = cast("Flight", self)

        if not isinstance(points, list):
            points = [points]

        return min(
            (cp(self.data, point) for point in points),
            key=attrgetter("distance"),
        )

    def takeoff_airport(self, **kwargs) -> "Airport":
        """Returns the most probable takeoff airport.

        .. code:: python

            >>> belevingsvlucht.takeoff_airport()
            EHAM/AMS: Amsterdam  Schiphol

        When data is missing near the ground, it may be relevant
        to specify a subset of airports as a keyword parameter.

        .. code:: python

            >>> missing_data = belevingsvlucht.after("2018-05-30 15:30")
            >>> missing_data.takeoff_airport()
            NL-0015/nan: Universitair Medisch Centrum Utrecht Heliport

            >>> large_airports = airports.query("type == 'large_airport'")
            >>> missing_data.takeoff_airport(dataset=large_airports)
            EHAM/AMS: Amsterdam  Schiphol
        """

        from ..core.distance import guess_airport

        # The following cast secures the typing
        self = cast("Flight", self)

        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[0], **kwargs)

    def landing_airport(self, **kwargs) -> "Airport":
        """Returns the most probable landing airport.

        .. code:: python

            >>> belevingsvlucht.landing_airport()
            EHAM/AMS: Amsterdam  Schiphol

        When data is missing near the ground, it may be relevant
        to specify a subset of airports as a keyword parameter.

        .. code:: python

            >>> missing_data = belevingsvlucht.before("2018-05-30 20:00")
            >>> missing_data.landing_airport()
            NL-0024/nan: Middenmeer Aerodrome

            >>> large_airports = airports.query("type == 'large_airport'")
            >>> missing_data.landing_airport(dataset=large_airports)
            EHAM/AMS: Amsterdam  Schiphol
        """

        from ..core.distance import guess_airport

        # The following cast secures the typing
        self = cast("Flight", self)

        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[-1], **kwargs)

    def aligned_on_runway(
        self, airport: Union[str, "Airport"]
    ) -> Iterator["Flight"]:
        """Iterates on all segments of trajectory matching a runway of the
        given airport.

        Example usage:

        >>> sum(1 for _ in belevingsvlucht.aligned_on_runway("EHAM"))
        2
        """

        from ..data import airports

        # The following cast secures the typing
        self = cast("Flight", self)

        _airport = airports[airport] if isinstance(airport, str) else airport
        if _airport is None or _airport.runways.shape.is_empty:
            return None

        if isinstance(_airport.runways.shape, LineString):
            candidate_shapes = [
                LineString(list(self.xy_time)).intersection(
                    _airport.runways.shape.buffer(5e-4)
                )
            ]
        else:
            candidate_shapes = [
                LineString(list(self.xy_time)).intersection(
                    on_runway.buffer(5e-4)
                )
                for on_runway in _airport.runways.shape
            ]

        for intersection in candidate_shapes:
            if intersection.is_empty:
                continue
            if isinstance(intersection, LineString):
                (*_, start), *_, (*_, stop) = intersection.coords
                segment = self.between(start, stop, strict=False)
                if segment is not None:
                    yield segment
            if isinstance(intersection, MultiLineString):
                (*_, start), *_, (*_, stop) = intersection[0].coords
                for chunk in intersection:
                    (*_, start_bak), *_, (*_, stop) = chunk.coords
                    if stop - start > 40:  # crossing runways and back
                        start = start_bak
                segment = self.between(start, stop, strict=False)
                if segment is not None:
                    yield segment

    def on_runway(self, airport: Union[str, "Airport"]) -> Optional["Flight"]:
        """Returns the longest segment of trajectory which perfectly matches
        a runway at given airport.

        .. code:: python

            >>> landing = belevingsvlucht.last(minutes=30).on_runway("EHAM")
            >>> landing.mean("altitude")
            -26.0

            >>> takeoff = belevingsvlucht.first(minutes=30).on_runway("EHAM")
            >>> takeoff.mean("altitude")
            437.27272727272725

        """

        return max(
            self.aligned_on_runway(airport),
            key=attrgetter("duration"),
            default=None,
        )

    def aligned_on_ils(
        # TODO None may not be a so good idea
        self,
        airport: Union[None, str, "Airport"],
    ) -> Iterator["Flight"]:
        """Iterates on all segments of trajectory aligned with the ILS of the
        given airport. The runway number is appended as a new ``ILS`` column.

        Example usage:

        .. code:: python

            >>> aligned = next(belevingsvlucht.aligned_on_ils('EHAM'))
            >>> f"ILS {aligned.max('ILS')} until {aligned.stop:%H:%M}"
            'ILS 06 until 20:17'

        Be aware that all segments are not necessarily yielded in order.
        Consider using ``max(..., key=attrgetter('start'))`` if you want the
        last landing attempt, or ``sorted(..., key=attrgetter('start'))`` for
        an ordered list

        .. code:: python

            >>> for aligned in belevingsvlucht.aligned_on_ils('EHLE'):
            ...     print(aligned.start)
            2018-05-30 16:50:44+00:00
            2018-05-30 18:13:02+00:00
            2018-05-30 16:00:55+00:00
            2018-05-30 17:21:17+00:00
            2018-05-30 19:05:22+00:00
            2018-05-30 19:42:36+00:00

            >>> from operator import attrgetter
            >>> last_aligned = max(
            ...     belevingsvlucht.aligned_on_ils("EHLE"),
            ...     key=attrgetter('start')
            ... )
        """

        from ..data import airports

        # The following cast secures the typing
        self = cast("Flight", self)

        if airport is None:
            airport = self.landing_airport()

        _airport = airports[airport] if isinstance(airport, str) else airport
        if _airport is None or _airport.runways.shape.is_empty:
            return None

        rad = np.pi / 180

        for threshold in _airport.runways.list:
            tentative = (
                self.bearing(threshold)
                .distance(threshold)
                .assign(
                    b_diff=lambda df: df.distance
                    * (np.radians(df.bearing - threshold.bearing).abs())
                )
                .query(
                    f"b_diff.abs() < .1 and cos((bearing - track) * {rad}) > 0"
                )
            )
            if tentative is not None:
                for chunk in tentative.split("20s"):
                    if chunk.longer_than("1 minute"):
                        yield chunk.assign(ILS=threshold.name)

    def aligned_on_navpoint(
        self,
        points: Iterable["Navaid"],
        angle_precision: int = 1,
        time_precision: str = "2T",
        min_time: str = "30s",
        min_distance: int = 80,
    ) -> Iterator["Flight"]:

        # The following cast secures the typing
        self = cast("Flight", self)
        for navpoint in points:
            tentative = (
                self.distance(navpoint)
                .bearing(navpoint)
                .assign(
                    b_diff=lambda df: df.distance
                    * (np.radians(df.bearing - df.track).abs()),
                    delta=lambda df: (df.bearing - df.track),
                )
                .query(f"delta.abs() < {angle_precision} and distance < 500")
            )
            if tentative is not None:
                for chunk in tentative.split(time_precision):
                    if (
                        chunk.longer_than(min_time)
                        and chunk.min("distance") < min_distance
                    ):
                        yield chunk.assign(navaid=navpoint.name)

    def self_intersections(self) -> int:
        # documentation TODO
        simplified: "Flight" = self.simplify(25)  # type: ignore
        if simplified.shape is None:
            return -1
        return len(simplified.shape.buffer(1e-3).interiors)

    def holding_pattern(
        self,
        min_altitude=7000,
        turning_threshold=.5,
        low_limit=pd.Timedelta("30 seconds"),
        high_limit=pd.Timedelta("10 minutes"),
        turning_limit = pd.Timedelta("5 minutes"),
    ) -> Iterator["Flight"]:
        # documentation TODO
        # thresholds (set arguments? TODO)

        # avoid parts that are really way too low
        alt_above = self.query(f"altitude > {min_altitude}")
        if alt_above is None:
            return

        straight_line = (
            alt_above.assign(
                turning_rate=lambda x: x.track_unwrapped.diff()
                / x.timestamp.diff().dt.total_seconds()
            )
            .filter(turning_rate=17)
            .query(f"turning_rate.abs() < {turning_threshold}")
        )
        if straight_line is None:
            return

        chunk_candidates = list(
            (chunk.start, chunk.duration, chunk.mean("track_unwrapped"), chunk)
            for chunk in straight_line.split("10s")
            if low_limit <= chunk.duration < high_limit
        )

        next_ = None
        for (
            (start1, duration1, track1, chunk1),
            (start2, _, track2, chunk2),
        ) in zip(chunk_candidates, chunk_candidates[1:]):
            if (
                start2 - start1 - duration1 < turning_limit
                and abs(abs(track1 - track2) - 180) < 15
            ):
                yield chunk1
                next_ = chunk2
            else:
                if next_ is not None:
                    yield next_
                next_ = None
