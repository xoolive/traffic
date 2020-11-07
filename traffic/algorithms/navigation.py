# fmt: off

from operator import attrgetter
from typing import (
    TYPE_CHECKING, Iterable, Iterator, List, Optional, Union, cast
)

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from ..core.iterator import flight_iterator

if TYPE_CHECKING:
    from ..core import Flight  # noqa: 401
    from ..core.mixins import PointMixin  # noqa: 401
    from ..core.structure import Airport, Navaid  # noqa: 401
    from ..data.basic.airports import Airports  # noqa: 401

# fmt: on


class NavigationFeatures:

    # shape: Optional[LineString]
    # query: Callable[["NavigationFeatures", str], Optional["Flight"]]

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

    def landing_at(self, airport: Union[str, "Airport"]) -> bool:
        from ..core.structure import Airport
        from ..data import airports

        return self.landing_airport() == (
            airport if isinstance(airport, Airport) else airports[airport]
        )

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

    @flight_iterator
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

    @flight_iterator
    def aligned_on_ils(
        self, airport: Union[None, str, "Airport"],
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

        chunks = list()
        for threshold in _airport.runways.list:
            tentative = (
                self.bearing(threshold)
                .distance(threshold)
                .assign(
                    b_diff=lambda df: df.distance
                    * np.radians(df.bearing - threshold.bearing).abs()
                )
                .query(f"b_diff < .1 and cos((bearing - track) * {rad}) > 0")
            )
            if tentative is not None:
                for chunk in tentative.split("20s"):
                    if (
                        chunk.longer_than("1 minute")
                        and chunk.altitude_min < 5000
                    ):
                        chunks.append(
                            chunk.assign(
                                ILS=threshold.name, airport=_airport.icao
                            )
                        )

        yield from sorted(chunks, key=attrgetter("start"))

    @flight_iterator
    def aligned_on_navpoint(
        self,
        points: Iterable["Navaid"],
        angle_precision: int = 1,
        time_precision: str = "2T",
        min_time: str = "30s",
        min_distance: int = 80,
    ) -> Iterator["Flight"]:
        """Iterates on segments of trajectories aligned with one of the given
        navigational beacons passed in parameter.

        The name of the navigational beacon is assigned in a new column
        `navaid`.

        """

        # The following cast secures the typing
        self = cast("Flight", self)
        for navpoint in points:
            tentative = (
                self.distance(navpoint)
                .bearing(navpoint)
                .assign(
                    b_diff=lambda df: df.distance
                    * (np.radians(df.bearing - df.track).abs()),
                    delta=lambda df: (df.bearing - df.track).abs(),
                )
                .query(f"delta < {angle_precision} and distance < 500")
            )
            if tentative is not None:
                for chunk in tentative.split(time_precision):
                    if (
                        chunk.longer_than(min_time)
                        and chunk.min("distance") < min_distance
                    ):
                        yield chunk.assign(navaid=navpoint.name)

    @flight_iterator
    def emergency(self) -> Iterator["Flight"]:
        """Iterates on emergency segments of trajectory.

        An emergency is defined with a 7700 squawk code.
        """
        sq7700 = self.query("squawk == '7700'")  # type: ignore
        if sq7700 is None:
            return
        yield from sq7700.split()

    @flight_iterator
    def runway_change(
        self,
        airport: Union[str, "Airport", None] = None,
        dataset: Optional["Airports"] = None,
    ) -> Iterator["Flight"]:
        """Detects runway changes.

        The method yields pieces of trajectories with exactly two runway
        alignments on the same airport not separated by a climbing phase.

        In each piece of yielded trajectory, the `ILS` column contains the
        name of the runway targetted by the aircraft at each instant.
        """

        # The following cast secures the typing
        self = cast("Flight", self)

        if airport is None:
            if dataset is None:
                airport = self.landing_airport()
            else:
                airport = self.landing_airport(dataset=dataset)

        if airport is None:
            return None

        aligned = iter(self.aligned_on_ils(airport))
        first = next(aligned, None)
        if first is None:
            return

        for second in aligned:
            candidate = self.between(first.start, second.stop)
            assert candidate is not None
            candidate = candidate.assign(ILS=None)
            if candidate.phases().query('phase == "CLIMB"') is None:
                candidate.data.loc[
                    candidate.data.timestamp <= first.stop, "ILS"
                ] = first.max("ILS")
                candidate.data.loc[
                    candidate.data.timestamp >= second.start, "ILS"
                ] = second.max("ILS")

                yield candidate.assign(
                    airport=airport
                    if isinstance(airport, str)
                    else airport.icao
                )

            first = second

    @flight_iterator
    def go_around(
        self,
        airport: Union[str, "Airport", None] = None,
        dataset: Optional["Airports"] = None,
    ) -> Iterator["Flight"]:
        """Detects go-arounds.

        The method yields pieces of trajectories with exactly two landing
        attempts (aligned on one runway) on the same airport separated by
        exactly one climbing phase.
        """

        # The following cast secures the typing
        self = cast("Flight", self)

        if airport is None:
            if dataset is None:
                airport = self.landing_airport()
            else:
                airport = self.landing_airport(dataset=dataset)

        if airport is None:
            return None

        first_attempt = next(self.aligned_on_ils(airport), None)

        while first_attempt is not None:
            after_first_attempt = self.after(first_attempt.start)
            assert after_first_attempt is not None

            climb = after_first_attempt.phases().query('phase == "CLIMB"')
            if climb is None:
                return

            after_climb = self.after(next(climb.split("10T")).stop)
            if after_climb is None:
                return

            next_attempt = next(after_climb.aligned_on_ils(airport), None)

            if next_attempt is not None:
                goaround = self.between(first_attempt.start, next_attempt.stop)
                assert goaround is not None

                goaround = goaround.assign(
                    ILS=None,
                    airport=airport
                    if isinstance(airport, str)
                    else airport.icao,
                )
                goaround.data.loc[
                    goaround.data.timestamp <= first_attempt.stop, "ILS"
                ] = first_attempt.max("ILS")
                goaround.data.loc[
                    goaround.data.timestamp >= next_attempt.start, "ILS"
                ] = next_attempt.max("ILS")
                yield goaround

            first_attempt = next_attempt

    @flight_iterator
    def landing_attempts(
        self, dataset: Optional["Airports"] = None
    ) -> Iterator["Flight"]:
        """Iterates on all landing attempts for current flight.

        First, candidates airports are identified in the neighbourhood
        of the segments of trajectory below 10,000 ft. By default, the
        full airport database is considered but it is possible to restrict
        it and pass a smaller database with the dataset parameter.

        If no runway information is available for the given airport, no
        trajectory segment will be provided.

        .. warning::

            This API is not stable yet. The interface may change in a near
            future.

        """
        candidate = self.query("altitude < 8000")  # type: ignore
        if candidate is not None:
            for chunk in candidate.split("10T"):
                point = chunk.query("altitude == altitude.min()")
                if dataset is None:
                    cd = point.landing_airport()
                else:
                    cd = point.landing_airport(dataset=dataset)
                if cd.runways is not None:
                    yield from chunk.assign(airport=cd.icao).aligned_on_ils(cd)

    def diversion(self) -> Optional["Flight"]:
        """Returns the segment of trajectory after a possible decision
        of diversion.

        The method relies on the `destination` parameter to identify the
        intended destination.

        """
        from ..data import airports

        f_above = self.query("altitude > 15000")  # type: ignore
        if (
            self.destination != self.destination  # type: ignore
            or airports[self.destination] is None  # type: ignore
            or f_above is None
        ):
            return None

        return (
            f_above.distance(airports[self.destination])  # type: ignore
            .diff("distance")
            .agg_time("10T", distance_diff="mean")
            .query("distance_diff > 0")
        )

    def diversion_ts(self) -> pd.Timestamp:
        diversion = self.diversion()
        if diversion is None:
            return pd.Timestamp("NaT")
        return diversion.start

    @property
    def holes(self) -> int:
        """Returns the number of 'holes' in a trajectory."""
        simplified: "Flight" = self.simplify(25)  # type: ignore
        if simplified.shape is None:
            return -1
        return len(simplified.shape.buffer(1e-3).interiors)

    @flight_iterator
    def holding_pattern(
        self,
        min_altitude=7000,
        turning_threshold=0.5,
        low_limit=pd.Timedelta("30 seconds"),  # noqa: B008
        high_limit=pd.Timedelta("10 minutes"),  # noqa: B008
        turning_limit=pd.Timedelta("5 minutes"),  # noqa: B008
    ) -> Iterator["Flight"]:
        """Iterates on parallel segments candidates for identifying
        a holding pattern.

        .. warning::

            This API is not stable yet. The interface may change in a near
            future.

        """
        # avoid parts that are really way too low
        alt_above = self.query(f"altitude > {min_altitude}")  # type: ignore
        if alt_above is None:
            return

        straight_line = (
            alt_above.unwrap()
            .assign(
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
