from typing import Iterator

import pitot.geodesy as geo

from shapely import LineString, Point
from shapely.geometry.base import BaseGeometry

from ...core import Flight
from ...core.structure import Airport
from ...core.time import deltalike, to_timedelta


class StartMoving:
    """
    Returns the part of the trajectory after the aircraft starts moving.

    .. warning::

        This method has been extensively tested on aircraft taxiing before
        take-off. It should be adapted/taken with extra care for
        trajectories after landing.

    """

    def __init__(
        self,
        speed_threshold: float = 2,
        time_threshold: str = "30s",
        filter_dict: dict[str, int] = dict(compute_gs=3),
        resample_rule: str = "5s",
    ):
        self.speed_threshold = speed_threshold
        self.time_threshold = time_threshold
        self.filter_dict = filter_dict
        self.resample_rule = resample_rule

    def apply(self, flight: Flight) -> None | Flight:
        resampled = flight.resample(self.resample_rule)
        if resampled is None or len(resampled) <= 2:
            return None

        moving = (
            resampled.cumulative_distance()
            .filter(**self.filter_dict)  # type: ignore
            .query(f"compute_gs > {self.speed_threshold}")
        )
        if moving is None:
            return None

        segment = None
        first_segment = None
        for segment in moving.split("1 min"):
            if (
                segment.longer_than(self.time_threshold)
                and first_segment is None
            ):
                first_segment = segment
        last_segment = segment

        if first_segment is None or last_segment is None:
            return None

        return flight.between(first_segment.start, last_segment.stop)


def mrr_diagonal(geom: BaseGeometry) -> float:
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


class Deprecated:  # TODO
    def slow_taxi(
        self,
        flight: Flight,
        min_duration: deltalike = "60s",
        max_diameter: float = 150,  # in meters
    ) -> Iterator["Flight"]:
        """
        Holding segments are part of a trajectory where the aircraft stays more
        than min_duration (in s) within a circle of diameter max_diameter (in m)

        TODO that method is not integrated in Flight,
        let's think whether it is really used/necessary
        """

        duration_threshold = to_timedelta(min_duration)

        current_flight = flight.movement()
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
                        candidate = flight.between(segment_begin, segment_end)
                        if candidate is not None:
                            yield candidate
                        segment_geoms = []
                        segment_times = []

            previously_stopped = is_stopped

        if (
            is_stopped
            and segment_times[-1] - segment_times[0] >= duration_threshold
        ):
            candidate = flight.between(segment_times[0], segment_times[-1])
            if candidate is not None:
                yield candidate

    def ground_trajectory(
        self, flight: Flight, airport: str | Airport
    ) -> Iterator[Flight]:
        """Returns the ground part of the trajectory limited to the apron
        of the airport passed in parameter.

        The same trajectory could use the apron several times, hence the safest
        option to return a FlightIterator.

        TODO that method is not integrated in Flight,
        let's think whether it is really used/necessary
        it is also very slow
        """

        from traffic.data import airports

        airport_ = airports[airport] if isinstance(airport, str) else airport
        assert airport_ is not None

        has_onground = "onground" in flight.data.columns
        criterion = "altitude < 5000"
        if has_onground:
            criterion += " or onground"

        low_altitude = flight.query(criterion)
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
