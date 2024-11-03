from __future__ import annotations

import logging
import warnings
from operator import attrgetter
from pathlib import Path
from pkgutil import get_data
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import pitot.geodesy as geo
from typing_extensions import NotRequired, TypedDict

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, Polygon, base

from ..core.iterator import flight_iterator
from ..core.time import deltalike, to_timedelta

if TYPE_CHECKING:
    from cartes.osm import Overpass

    from ..core import Flight, FlightPlan
    from ..core.mixins import PointMixin
    from ..core.structure import Airport, Navaid
    from ..data.basic.airports import Airports
    from ..data.basic.navaid import Navaids


_log = logging.getLogger(__name__)


def mrr_diagonal(geom: base.BaseGeometry) -> float:
    """
    Returns the length of the diagonal of the minimum rotated rectangle around
    a given shape.

    Consider using a :meth:`~traffic.core.mixins.ShapelyMixin.project_shape`
    method before applying this method if you need a distance in meters.

    """
    if len(geom) <= 1:
        return 0
    if len(geom) == 2:
        return geo.distance(  # type: ignore
            lat1=geom[0].y, lon1=geom[0].x, lat2=geom[1].y, lon2=geom[1].x
        )
    mrr = LineString(geom).minimum_rotated_rectangle
    if isinstance(mrr, Point):
        return 0
    try:  # in most cases, mrr is a Polygon
        x, y = mrr.exterior.coords.xy
    except AttributeError:  # then it should be a LineString
        p0, p1 = mrr.coords[0], mrr.coords[-1]
        return geo.distance(p0[1], p0[0], p1[1], p1[0])  # type: ignore
    return geo.distance(y[0], x[0], y[2], x[2])  # type: ignore


class PointMergeParams(TypedDict):
    point_merge: str | PointMixin
    secondary_point: NotRequired[None | str | PointMixin]
    distance_interval: NotRequired[tuple[float, float]]
    delta_threshold: NotRequired[float]


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

    # -- Most basic metadata properties --

    def takeoff_from(self, airport: Union[str, "Airport"]) -> bool:
        """Returns True if the flight takes off from the given airport."""

        from ..core.structure import Airport
        from ..data import airports

        return self.takeoff_airport() == (
            airport if isinstance(airport, Airport) else airports[airport]
        )

    def takeoff_airport(self, **kwargs: Any) -> "Airport":
        """Returns the most probable takeoff airport based on the first location
        in the trajectory.

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
        """Returns True if the flight lands at the given airport."""

        from ..core.structure import Airport
        from ..data import airports

        return self.landing_airport() == (
            airport if isinstance(airport, Airport) else airports[airport]
        )

    def landing_airport(self, **kwargs: Any) -> "Airport":
        """Returns the most probable landing airport based on the last location
        in the trajectory.

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

    # -- Alignments --

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
        if (
            _airport is None
            or _airport.runways is None
            or _airport.runways.shape.is_empty
        ):
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
                for on_runway in _airport.runways.shape.geoms
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
                (*_, start), *_, (*_, stop) = intersection.geoms[0].coords
                for chunk in intersection.geoms:
                    (*_, start_bak), *_, (*_, stop) = chunk.coords
                    if stop - start > 40:  # crossing runways and back
                        start = start_bak
                segment = self.between(start, stop, strict=False)
                if segment is not None:
                    yield segment

    # >>>>> This function is to be deprecated

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
        msg = "Use .aligned_on_runway(airport).max() instead."
        warnings.warn(msg, DeprecationWarning)

        return max(
            self.aligned_on_runway(airport),
            key=attrgetter("duration"),
            default=None,
        )

    # --- end ---

    @flight_iterator
    def aligned_on_ils(
        self,
        airport: Union[None, str, "Airport"],
        angle_tolerance: float = 0.1,
        min_duration: deltalike = "1 min",
    ) -> Iterator["Flight"]:
        """Iterates on all segments of trajectory aligned with the ILS of the
        given airport. The runway number is appended as a new ``ILS`` column.

        :param airport: Airport where the ILS is located
        :param angle_tolerance: maximum tolerance on bearing difference between
            ILS and flight trajectory.
        :param min_duration: minimum duration a flight has to spend on the ILS
            to be considered as aligned.

        Example usage:

        .. code:: python

            >>> aligned = belevingsvlucht.aligned_on_ils('EHAM').next()
            >>> f"ILS {aligned.max('ILS')} until {aligned.stop:%H:%M}"
            'ILS 06 until 20:17'

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
        if (
            _airport is None
            or _airport.runways is None
            or _airport.runways.shape.is_empty
        ):
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
                .query(
                    f"b_diff < {angle_tolerance} and "
                    f"cos((bearing - track) * {rad}) > 0"
                )
            )
            if tentative is not None:
                for chunk in tentative.split("20s"):
                    if (
                        chunk.longer_than(min_duration)
                        and not pd.isna(altmin := chunk.altitude_min)
                        and altmin < 5000
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
        points: Union[str, "PointMixin", Iterable["PointMixin"], "FlightPlan"],
        angle_precision: int = 1,
        time_precision: str = "2 min",
        min_time: str = "30s",
        min_distance: int = 80,
    ) -> Iterator["Flight"]:
        """Iterates on segments of trajectories aligned with one of the given
        navigational beacons passed in parameter.

        The name of the navigational beacon is assigned in a new column
        `navaid`.

        """

        from ..core import FlightPlan
        from ..core.mixins import PointMixin

        points_: Sequence[PointMixin]

        # The following cast secures the typing
        self = cast("Flight", self)

        if isinstance(points, str):
            from ..data import navaids

            navaid = navaids[points]
            if navaid is None:
                _log.warning(f"Navaid {points} unknown")
                points_ = []
            else:
                points_ = [navaid]
        elif isinstance(points, PointMixin):
            points_ = [points]
        elif isinstance(points, FlightPlan):
            points_ = points.all_points
        else:
            points_ = list(points)

        for navpoint in points_:
            tentative = (
                self.distance(navpoint)
                .bearing(navpoint)
                .assign(
                    shift=lambda df: df.distance
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

    def compute_navpoints(
        self, navaids: Optional["Navaids"] = None, buffer: float = 0.1
    ) -> Optional[pd.DataFrame]:
        """This functions recomputes the most probable alignments on
        navigational points on the trajectory.

        By default, all navaids of the default database are considered,
        but limited to a buffered bounding box around the trajectory.

        Once computed, the following Altair snippet may be useful to
        display the trajectory as a succession of segments:

        .. code:: python

            import altair as alt

            df = flight.compute_navpoints()

            segments = (
                alt.Chart(df.drop(columns="duration")).encode(
                    alt.X("start", title=None),
                    alt.X2("stop"),
                    alt.Y("navaid", sort="x", title=None),
                    alt.Color("type", title="Navigational point"),
                    alt.Tooltip(["navaid", "distance", "shift_mean"]),
                )
                .mark_bar(size=10)
                .configure_legend(
                    orient="bottom",
                    labelFontSize=14, titleFontSize=14, labelFont="Ubuntu"
                )
                .configure_axis(labelFontSize=14, labelFont="Ubuntu")
            )

        """

        # The following cast secures the typing
        self = cast("Flight", self)

        if navaids is None:
            from ..data import navaids as default_navaids

            navaids = default_navaids

        navaids_ = navaids.extent(self, buffer=buffer)
        if navaids_ is None:
            return None
        navaids_ = navaids_.drop_duplicates("name")
        all_points: List["Navaid"] = list(navaids_)

        def all_aligned_segments(traj: "Flight") -> pd.DataFrame:
            return pd.DataFrame.from_records(
                list(
                    {
                        "start": segment.start,
                        "stop": segment.stop,
                        "duration": segment.duration,
                        "navaid": segment.max("navaid"),
                        "distance": segment.min("distance"),
                        "shift_mean": segment.shift_mean,
                        "shift_meanp": segment.shift_mean + 0.02,
                    }
                    for segment in traj.aligned_on_navpoint(all_points)
                )
            ).sort_values("start")

        def groupby_intervals(table: pd.DataFrame) -> Iterator[pd.DataFrame]:
            if table.shape[0] == 0:
                return
            table = table.sort_values("start")
            # take as much as you can
            sweeping_line = table.query("stop <= stop.iloc[0]")
            # try to push the stop line: which intervals overlap the stop line?
            additional = table.query(
                "start <= @sweeping_line.stop.max() < stop"
            )

            while additional.shape[0] > 0:
                sweeping_line = table.query("stop <= @additional.stop.max()")
                additional = table.query(
                    "start <= @sweeping_line.stop.max() < stop"
                )

            yield sweeping_line
            yield from groupby_intervals(
                table.query("start > @sweeping_line.stop.max()")
            )

        def most_probable_navpoints(traj: "Flight") -> Iterator[pd.DataFrame]:
            table = all_aligned_segments(traj)
            for block in groupby_intervals(table):
                d_max = block.eval("duration.max()")
                t_threshold = d_max - pd.Timedelta("30s")  # noqa: F841
                yield (
                    block.sort_values("shift_mean")
                    .query("duration >= @t_threshold")
                    .head(1)
                )

        return pd.concat(list(most_probable_navpoints(self))).merge(
            navaids_.data, left_on="navaid", right_on="name"
        )

    @flight_iterator
    def takeoff_from_runway(
        self,
        airport: Union[str, "Airport"],
        threshold_alt: int = 2000,
        zone_length: int = 6000,
        little_base: int = 50,
        opening: float = 5,
    ) -> Iterator["Flight"]:
        """Identifies the take-off runway for trajectories.

        Iterates on all segments of trajectory matching a zone around a runway
        of the  given airport. The takeoff runway number is appended as a new
        ``runway`` column.

        """

        from ..data import airports

        # Access completion on Flight objects
        self = cast("Flight", self).phases()

        _airport = airports[airport] if isinstance(airport, str) else airport
        if (
            _airport is None
            or _airport.runways is None
            or _airport.runways.shape.is_empty
        ):
            return None

        nb_run = len(_airport.runways.data)
        alt = _airport.altitude + threshold_alt
        base = zone_length * np.tan(opening * np.pi / 180) + little_base

        # Create shapes around each runway
        list_p0 = geo.destination(
            list(_airport.runways.data.latitude),
            list(_airport.runways.data.longitude),
            list(_airport.runways.data.bearing),
            [zone_length for i in range(nb_run)],
        )
        list_p1 = geo.destination(
            list(_airport.runways.data.latitude),
            list(_airport.runways.data.longitude),
            [x + 90 for x in list(_airport.runways.data.bearing)],
            [little_base for i in range(nb_run)],
        )
        list_p2 = geo.destination(
            list(_airport.runways.data.latitude),
            list(_airport.runways.data.longitude),
            [x - 90 for x in list(_airport.runways.data.bearing)],
            [little_base for i in range(nb_run)],
        )
        list_p3 = geo.destination(
            list_p0[0],
            list_p0[1],
            [x - 90 for x in list(_airport.runways.data.bearing)],
            [base for i in range(nb_run)],
        )
        list_p4 = geo.destination(
            list_p0[0],
            list_p0[1],
            [x + 90 for x in list(_airport.runways.data.bearing)],
            [base for i in range(nb_run)],
        )

        runway_polygons = {}

        for i, name in enumerate(_airport.runways.data.name):
            lat = [list_p1[0][i], list_p2[0][i], list_p3[0][i], list_p4[0][i]]
            lon = [list_p1[1][i], list_p2[1][i], list_p3[1][i], list_p4[1][i]]

            poly = Polygon(zip(lon, lat))
            runway_polygons[name] = poly

        low_traj = self.query(
            f"(phase == 'CLIMB' or phase == 'LEVEL') and altitude < {alt}"
        )

        if low_traj is None:
            return

        for segment in low_traj.split("2 min"):
            candidates_set = []
            for name, polygon in runway_polygons.items():
                if segment.intersects(polygon):
                    candidate = (
                        segment.cumulative_distance()
                        .clip_iterate(polygon)
                        .max(key="compute_gs_max")
                    )
                    if candidate is None or candidate.shape is None:
                        continue
                    start_runway = candidate.aligned_on_runway(airport).max()

                    if start_runway is not None:
                        candidate = candidate.after(start_runway.start)
                        if candidate is None or candidate.shape is None:
                            continue
                    if candidate.max("compute_gs") < 140:
                        continue

                    candidates_set.append(candidate.assign(runway=name))

            result = max(
                candidates_set, key=attrgetter("duration"), default=None
            )
            if result is not None:
                yield result

    # -- Special operations --

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
        **kwargs: Any,
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

        aligned = iter(self.aligned_on_ils(airport, **kwargs))
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
        airport: None | str | "Airport" = None,
        dataset: None | "Airports" = None,
        **kwargs: Any,
    ) -> Iterator["Flight"]:
        """Detects go-arounds.

        The method yields pieces of trajectories with exactly two landing
        attempts (aligned on one runway) on the same airport separated by
        exactly one climbing phase.

        :param airport: If None, the method tries to guess the landing airport
            based on the ``dataset`` parameter. (see
            :meth:`~traffic.core.Flight.landing_airport`)
        :param dataset: database of candidate airports, only used if ``airport``
            is None

        **See also:** :ref:`How to select go-arounds from a set of
        trajectories?`
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

        attempts = self.aligned_on_ils(airport, **kwargs)
        # you need to be aligned at least twice on a rway to have a GA:
        if len(attempts) < 2:
            return

        first_attempt = next(attempts, None)

        while first_attempt is not None:
            after_first_attempt = self.after(first_attempt.start)
            assert after_first_attempt is not None

            climb = after_first_attempt.phases().query('phase == "CLIMB"')
            if climb is None:
                return

            after_climb = self.after(next(climb.split("10 min")).stop)
            if after_climb is None:
                return

            next_attempt = next(
                after_climb.aligned_on_ils(airport, **kwargs), None
            )

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
        self, dataset: Optional["Airports"] = None, **kwargs: Any
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
        # The following cast secures the typing
        self = cast("Flight", self)

        candidate = self.query("altitude < 8000")
        if candidate is not None:
            for chunk in candidate.split("10 min"):
                point = chunk.query("altitude == altitude.min()")
                if point is None:
                    return
                if dataset is None:
                    cd = point.landing_airport()
                else:
                    cd = point.landing_airport(dataset=dataset)
                if cd.runways is not None:
                    yield from chunk.assign(airport=cd.icao).aligned_on_ils(
                        cd, **kwargs
                    )

    def diversion(self) -> Optional["Flight"]:
        """Returns the segment of trajectory after a possible decision
        of diversion.

        The method relies on the `destination` parameter to identify the
        intended destination.

        """
        from ..data import airports

        # The following cast secures the typing
        self = cast("Flight", self)

        f_above: Optional["Flight"] = self.query("altitude > 15000")
        if (
            pd.isna(self.destination)
            or airports[self.destination] is None  # type: ignore
            or f_above is None
        ):
            return None

        return (
            f_above.distance(airports[self.destination])  # type: ignore
            .diff("distance")
            .agg_time("10 min", distance_diff="mean")
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

        # The following cast secures the typing
        self = cast("Flight", self)

        simplified = self.simplify(25)
        if simplified.shape is None:
            return -1
        return len(simplified.shape.buffer(1e-3).interiors)

    @flight_iterator
    def point_merge(
        self,
        point_merge: str | PointMixin | list[PointMergeParams],
        secondary_point: None | str | PointMixin = None,
        distance_interval: None | tuple[float, float] = None,
        delta_threshold: float = 5e-2,
        airport: None | str | Airport = None,
        runway: None | str = None,
        **kwargs: Any,
    ) -> Iterator["Flight"]:
        """
        Iterates on all point merge segments in a trajectory before landing at
        a given airport.

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

        (new in version 2.8)
        """
        # The following cast secures the typing
        self = cast("Flight", self)

        if isinstance(point_merge, list):
            results = []
            for params in point_merge:
                id_ = params.get("secondary_point", params["point_merge"])
                assert id_ is not None
                name = id_ if isinstance(id_, str) else id_.name
                for segment in self.point_merge(**params):
                    results.append(segment.assign(point_merge=name))
            yield from sorted(results, key=attrgetter("start"))
            return

        from traffic.data import navaids

        navaids_extent = navaids.extent(self, buffer=1)
        msg = f"No navaid available in the bounding box of Flight {self}"

        if isinstance(point_merge, str):
            if navaids_extent is None:
                _log.warn(msg)
                return None
            point_merge = navaids_extent.get(point_merge)  # type: ignore
            if point_merge is None:
                _log.warn("Navaid for point_merge not found")
                return None

        if secondary_point is None:
            secondary_point = point_merge

        if isinstance(secondary_point, str):
            if navaids_extent is None:
                _log.warn(msg)
                return None
            secondary_point = navaids_extent.get(secondary_point)
            if secondary_point is None:
                _log.warn("Navaid for secondary_point not found")
                return None

        if airport is not None:
            for landing in self.aligned_on_ils(airport, **kwargs):
                if runway is None or landing.max("ILS") == runway:
                    yield from self.point_merge(
                        point_merge=point_merge,
                        secondary_point=secondary_point,
                        distance_interval=distance_interval,
                        delta_threshold=delta_threshold,
                    )
            return

        for segment in self.aligned_on_navpoint(point_merge):
            before_point = self.before(segment.start)
            if before_point is None:
                continue
            before_point = before_point.last("10 minutes")
            if before_point is None:
                continue
            lower, upper = distance_interval if distance_interval else (0, 100)
            constant_distance = (
                before_point.distance(secondary_point)
                .diff("distance")
                .query(
                    f"{lower} < distance < {upper} and "
                    f"distance_diff.abs() < {delta_threshold}"
                )
            )
            if constant_distance is None:
                continue
            candidate = constant_distance.split("5 seconds").max()
            if candidate is not None and candidate.longer_than("90 seconds"):
                result = self.between(candidate.start, segment.stop)
                if result is not None:
                    yield result

    @flight_iterator
    def holding_pattern(
        self,
        duration: str = "6 min",
        step: str = "2 min",
        threshold: str = "5 min",
        samples: int = 30,
        model_path: None | str | Path = None,
        vertical_rate: bool = False,
    ) -> Iterator["Flight"]:
        """Iterates on all holding pattern segments in the trajectory.

        This approach is based on a neuronal network model. Details will be
        published in a coming academic publication.

        Parameters should be left as default as they are strongly coupled with
        the proposed model.

        The model has been trained on manually labelled holding patterns for
        trajectories landing at different European airports including London
        Heathrow.

        (new in version 2.8)
        """
        import onnxruntime as rt

        # The following cast secures the typing
        self = cast("Flight", self)

        providers = rt.get_available_providers()

        if model_path is None:
            pkg = "traffic.algorithms.onnx.holding_pattern"
            data = get_data(pkg, "scaler.onnx")
            scaler_sess = rt.InferenceSession(data, providers=providers)
            data = get_data(pkg, "classifier.onnx")
            classifier_sess = rt.InferenceSession(data, providers=providers)
        else:
            model_path = Path(model_path)
            scaler_sess = rt.InferenceSession(
                (model_path / "scaler.onnx").read_bytes(),
                providers=providers,
            )
            classifier_sess = rt.InferenceSession(
                (model_path / "classifier.onnx").read_bytes(),
                providers=providers,
            )

        start, stop = None, None

        for i, window in enumerate(self.sliding_windows(duration, step)):
            if window.duration >= pd.Timedelta(threshold):
                window = window.assign(flight_id=str(i))
                resampled = window.resample(samples)

                if resampled.data.eval("track.isnull()").any():
                    continue

                features = (
                    resampled.data.track_unwrapped
                    - resampled.data.track_unwrapped[0]
                ).values.reshape(1, -1)

                if vertical_rate:
                    if resampled.data.eval("vertical_rate.notnull()").any():
                        continue
                    vertical_rates = (
                        resampled.data.vertical_rate.values.reshape(1, -1)
                    )
                    features = np.concatenate(
                        (features, vertical_rates), axis=1
                    )

                name = scaler_sess.get_inputs()[0].name
                value = features.astype(np.float32)
                x = scaler_sess.run(None, {name: value})[0]

                name = classifier_sess.get_inputs()[0].name
                value = x.astype(np.float32)
                pred = classifier_sess.run(None, {name: value})[0]

                if bool(pred.round().item()):
                    if start is None:
                        start, stop = window.start, window.stop
                    elif start < stop:
                        stop = window.stop
                    else:
                        yield self.between(start, stop)
                        start, stop = window.start, window.stop
        if start is not None:
            yield self.between(start, stop)  # type: ignore

    # -- Airport ground operations specific methods --

    @flight_iterator
    def on_parking_position(
        self,
        airport: Union[str, "Airport"],
        buffer_size: float = 1e-5,  # degrees
        parking_positions: None | "Overpass" = None,
    ) -> Iterator["Flight"]:
        """
        Generates possible parking positions at a given airport.

        Example usage:

        >>> parking = flight.on_parking_position('LSZH').max()
        # returns the most probable parking position in terms of duration

        .. warning::

            This method has been well tested for aircraft taking off, but should
            be double checked for landing trajectories.

        """
        from ..data import airports

        # Access completion on Flight objects
        self = cast("Flight", self)

        _airport = airports[airport] if isinstance(airport, str) else airport
        if (
            _airport is None
            or _airport.runways is None
            or _airport.runways.shape.is_empty
        ):
            return None

        segment = self.filter().inside_bbox(_airport)
        if segment is None:
            return None

        segment = segment.split().max()

        parking_positions = (
            _airport.parking_position.query("type_ == 'way'")
            if parking_positions is None
            else parking_positions.query("type_ == 'way'")
        )
        for _, p in parking_positions.data.iterrows():
            if segment.intersects(p.geometry.buffer(buffer_size)):
                parking_part = segment.clip(p.geometry.buffer(buffer_size))
                if parking_part is not None:
                    yield parking_part.assign(parking_position=p.ref)

    def moving(
        self,
        speed_threshold: float = 2,
        time_threshold: str = "30s",
        filter_dict: Dict[str, int] = dict(compute_gs=3),
        resample_rule: str = "5s",
    ) -> Optional["Flight"]:
        """
        Returns the part of the trajectory after the aircraft starts moving.

        .. warning::

            This method has been extensively tested on aircraft taxiing before
            take-off. It should be adapted/taken with extra care for
            trajectories after landing.

        """

        self = cast("Flight", self)

        resampled = self.resample(resample_rule)
        if resampled is None or len(resampled) <= 2:
            return None

        moving = (
            resampled.cumulative_distance()
            .filter(**filter_dict)  # type: ignore
            .query(f"compute_gs > {speed_threshold}")
        )
        if moving is None:
            return None

        segment = None
        first_segment = None
        for segment in moving.split("1 min"):
            if segment.longer_than(time_threshold) and first_segment is None:
                first_segment = segment
        last_segment = segment

        if first_segment is None or last_segment is None:
            return None

        return self.between(first_segment.start, last_segment.stop)

    def pushback(
        self,
        airport: Union[str, "Airport"],
        filter_dict: Dict[str, int] = dict(
            compute_track_unwrapped=21, compute_track=21, compute_gs=21
        ),
        track_threshold: float = 90,
        parking_positions: None | "Overpass" = None,
    ) -> Optional["Flight"]:
        """
        Returns the pushback part of the trajectory on ground.

        The method identifies the start of the movement, the parking_position
        and the moment the aircraft suddenly changes direction the computed
        track angle.

        .. warning::

            The method has poor performance when trajectory point on ground are
            lacking. This is often the case for data recorded from locations far
            from the airport.

        """

        from ..data import airports

        # Access completion on Flight objects
        self = cast("Flight", self)

        _airport = airports[airport] if isinstance(airport, str) else airport
        if (
            _airport is None
            or _airport.runways is None
            or _airport.runways.shape.is_empty
        ):
            return None

        within_airport = self.inside_bbox(_airport)
        if within_airport is None:
            return None

        parking_position = within_airport.on_parking_position(
            _airport, parking_positions=parking_positions
        ).next()
        if parking_position is None:
            return None

        after_parking = within_airport.after(parking_position.start)
        assert after_parking is not None

        in_movement = cast(Optional["Flight"], after_parking.moving())

        if in_movement is None:
            return None

        # trim the first few seconds to avoid annoying first spike
        direction_change = (
            in_movement.first("5 min")
            .last("4 min 30s")
            .cumulative_distance()
            .unwrap(["compute_track"])
            .filter(**filter_dict)  # type: ignore
            .diff("compute_track_unwrapped")
            .query(f"compute_track_unwrapped_diff.abs() > {track_threshold}")
        )

        if direction_change is None:
            return None

        return in_movement.before(  # type: ignore
            direction_change.start
        ).assign(parking_position=parking_position.parking_position_max)

    def is_from_inertial(self, freq_threshold: float = 0.05) -> bool:
        """
        Returns True if ground trajectory data looks noisy.

        .. warning::

            This method is still experimental and tries to catch trajectories
            based on the inertial system rather than the GPS. It catches zig-zag
            patterns but fails to get trajectories driving between taxiways.

        """
        self = cast("Flight", self)

        if "compute_track" not in self.data.columns:
            self = self.cumulative_distance(compute_gs=False)

        freq = (
            self.diff("compute_track")
            .compute_track_diff.round()
            .value_counts(normalize=True)
        )
        if 90 not in freq.index or -90 not in freq.index:
            return False
        return (  # type: ignore
            freq[90] > freq_threshold and freq[-90] > freq_threshold
        )

    @flight_iterator
    def slow_taxi(
        self,
        min_duration: deltalike = "60s",
        max_diameter: float = 150,  # in meters
    ) -> Iterator["Flight"]:
        """
        Holding segments are part of a trajectory where the aircraft stays more
        than min_duration (in s) within a circle of diameter max_diameter (in m)

        """
        self = cast("Flight", self)

        duration_threshold = to_timedelta(min_duration)

        current_flight = self.moving()
        if current_flight is None:
            return None

        current_flight = current_flight.onground()
        if current_flight is None:
            return None

        current_flight = current_flight.resample("5s")
        if current_flight is None:
            return None

        traj_df = (
            current_flight.data[["timestamp", "latitude", "longitude"]]
            .sort_values("timestamp")
            .set_index("timestamp")
        )

        segment_geoms = []
        segment_times = []

        # Variables to detect changes between a stop
        # segment and a moving segment
        is_stopped = False
        previously_stopped = False

        # iterate over each coordinate to create segments
        # Each data point is added to a queue (FIFO)
        for index, row in traj_df.iterrows():
            segment_geoms.append(Point(row.longitude, row.latitude))
            segment_times.append(index)

            if not is_stopped:  # remove points to the specified min_duration
                while (
                    len(segment_geoms) > 2
                    and segment_times[-1] - segment_times[0]
                    >= duration_threshold
                ):
                    segment_geoms.pop(0)
                    segment_times.pop(0)

            # Check if current segment, trimmed to have a duration shorthen than
            # min_duration threshold,  is longer than the maximum distance
            # threshold
            if (
                len(segment_geoms) > 1
                and mrr_diagonal(segment_geoms) < max_diameter
            ):
                is_stopped = True
            else:
                is_stopped = False

            # detection of the end of a stop segment and append to
            # stop segment list
            if len(segment_geoms) > 1:
                segment_end = segment_times[-2]
                segment_begin = segment_times[0]
                if not is_stopped and previously_stopped:
                    if (
                        segment_end - segment_begin >= duration_threshold
                    ):  # detected end of a stop
                        candidate = self.between(segment_begin, segment_end)
                        if candidate is not None:
                            yield candidate
                        segment_geoms = []
                        segment_times = []

            previously_stopped = is_stopped

        if (
            is_stopped
            and segment_times[-1] - segment_times[0] >= duration_threshold
        ):
            candidate = self.between(segment_times[0], segment_times[-1])
            if candidate is not None:
                yield candidate

    @flight_iterator
    def ground_trajectory(
        self, airport: Union[str, "Airport"]
    ) -> Iterator["Flight"]:
        """Returns the ground part of the trajectory limited to the apron
        of the airport passed in parameter.

        The same trajectory could use the apron several times, hence the safest
        option to return a FlightIterator.
        """

        from traffic.data import airports

        self = cast("Flight", self)
        airport_ = airports[airport] if isinstance(airport, str) else airport
        assert airport_ is not None

        has_onground = "onground" in self.data.columns
        criterion = "altitude < 5000"
        if has_onground:
            criterion += " or onground"

        low_altitude = self.query(criterion)
        if low_altitude is None:
            return
        if airport_.shape is None:
            raise ValueError("No shape available for the given airport")
        for low_segment in low_altitude.split("10 min"):
            for airport_segment in low_segment.clip_iterate(
                airport_.shape.buffer(5e-3)
            ):
                if has_onground:
                    onground = airport_segment.query("onground")
                else:
                    onground = airport_segment.query(
                        "altitude < 500 or altitude.isnull()"
                    )
                if onground is not None:
                    yield onground

    @flight_iterator
    def on_taxiway(
        self,
        airport_or_taxiways: Union[str, pd.DataFrame, "Airport", "Overpass"],
        *,
        tolerance: float = 15,
        max_dist: float = 85,
    ) -> Iterator["Flight"]:
        """
        Iterates on segments of trajectory matching a single runway label.
        """

        from ..core.structure import Airport
        from ..data import airports

        self = cast("Flight", self)
        if isinstance(airport_or_taxiways, str):
            airport_or_taxiways = airports[airport_or_taxiways]
        # This is obvious, but not for MyPy
        assert not isinstance(airport_or_taxiways, str)

        taxiways_ = (
            airport_or_taxiways.taxiway
            if isinstance(airport_or_taxiways, Airport)
            else airport_or_taxiways
        )

        # decompose with a function because MyPy is lost
        def taxi_df(taxiways_: Union["Overpass", pd.DataFrame]) -> pd.DataFrame:
            if isinstance(taxiways_, pd.DataFrame):
                return taxiways_
            if taxiways_.data is None:
                raise ValueError("No taxiway information")
            return taxiways_.data

        taxiways = (  # one entry per runway label
            taxi_df(taxiways_)
            .groupby("ref")
            .agg({"geometry": list})["geometry"]
            .apply(MultiLineString)
            .to_frame()
        )

        simplified_df = cast(
            pd.DataFrame, self.simplify(tolerance=tolerance).data
        )
        if simplified_df.shape[0] < 2:
            return

        previous_candidate = None
        first = simplified_df.iloc[0]
        for _, second in simplified_df.iloc[1:].iterrows():
            p1 = Point(first.longitude, first.latitude)
            p2 = Point(second.longitude, second.latitude)

            def extremities_dist(twy: MultiLineString) -> float:
                p1_proj = twy.interpolate(twy.project(p1))
                p2_proj = twy.interpolate(twy.project(p2))
                d1 = geo.distance(p1_proj.y, p1_proj.x, p1.y, p1.x)
                d2 = geo.distance(p2_proj.y, p2_proj.x, p2.y, p2.x)
                return d1 + d2  # type: ignore

            temp_ = taxiways.assign(dist=np.vectorize(extremities_dist))
            start, stop, ref, dist = (
                first.timestamp,
                second.timestamp,
                temp_.dist.idxmin(),
                temp_.dist.min(),
            )
            if dist < max_dist:
                candidate = self.assign(taxiway=ref).between(start, stop)
                if previous_candidate is None:
                    previous_candidate = candidate

                else:
                    prev_ref = previous_candidate.taxiway_max
                    delta = start - previous_candidate.stop
                    if prev_ref == ref and delta < pd.Timedelta("1 min"):
                        previous_candidate = self.assign(taxiway=ref).between(
                            previous_candidate.start, stop
                        )

                    else:
                        yield previous_candidate
                        previous_candidate = candidate

            first = second

        if previous_candidate is not None:
            yield previous_candidate

    @flight_iterator
    def thermals(self) -> Iterator["Flight"]:
        """Detects thermals for gliders."""
        self = cast("Flight", self)
        all_segments = (
            self.unwrap()
            .diff("track_unwrapped")
            .agg_time(
                "1 min", vertical_rate="max", track_unwrapped_diff="median"
            )
            .abs(track_unwrapped_diff_median="track_unwrapped_diff_median")
            .query("vertical_rate_max > 2 and track_unwrapped_diff_median > 5")
        )
        if all_segments is not None:
            yield from all_segments.split("1 min")
