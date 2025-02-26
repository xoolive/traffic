from __future__ import annotations

from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

import pitot.geodesy as geo
from cartes.osm import Overpass

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, base

from ...core.iterator import flight_iterator
from ...core.time import deltalike, to_timedelta

if TYPE_CHECKING:
    from cartes.osm import Overpass

    from ...core import Flight
    from ...core.mixins import PointMixin
    from ...core.structure import Airport, Navaid
    from ...data.basic.airports import Airports
    from ...data.basic.navaid import Navaids


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
        from ...core.distance import closest_point as cp

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

        from ...core.structure import Airport
        from ...data import airports

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

        from ...core.distance import guess_airport

        # The following cast secures the typing
        self = cast("Flight", self)

        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[0], **kwargs)

    def landing_at(self, airport: Union[str, "Airport"]) -> bool:
        """Returns True if the flight lands at the given airport."""

        from ...core.structure import Airport
        from ...data import airports

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

        from ...core.distance import guess_airport

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

        from ...data import airports

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
            from ...data import navaids as default_navaids

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
        from ...data import airports

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

    # -- Airport ground operations specific methods --
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

        from ...core.structure import Airport
        from ...data import airports

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
