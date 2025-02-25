import logging
from typing import TYPE_CHECKING, Optional, Protocol, Union

import geopandas as gpd

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ...core.flight import Flight
from ..filters import FilterMedian

if TYPE_CHECKING:
    from cartes.osm import Overpass

    from shapely.geometry import Polygon

    from ...core.flight import Flight
    from ...core.structure import Airport


class PushbackBase(Protocol):
    def apply(self, flight: "Flight") -> Optional["Flight"]: ...


_log = logging.getLogger(__name__)


class ParkingPositionBasedPushback:
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

    def __init__(
        self,
        airport: Union[str, "Airport"],
        filter_dict: dict[str, int] = dict(
            compute_track_unwrapped=21, compute_track=21, compute_gs=21
        ),
        track_threshold: float = 90,
        parking_positions: Optional["Overpass"] = None,
    ) -> None:
        from ...data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if self.airport is None:
            raise RuntimeError("Airport information missing")

        self.filter_dict = filter_dict
        self.track_threshold = track_threshold
        self.parking_positions = parking_positions

    def apply(self, flight: "Flight") -> Optional["Flight"]:
        within_airport: Optional["Flight"] = flight.inside_bbox(self.airport)
        if within_airport is None:
            return None

        parking_position = within_airport.on_parking_position(
            self.airport, parking_positions=self.parking_positions
        ).next()
        if parking_position is None:
            return None

        after_parking = within_airport.after(parking_position.start)
        assert after_parking is not None

        in_movement = after_parking.moving()

        if in_movement is None:
            return None

        # trim the first few seconds to avoid annoying first spike
        direction_change = (
            in_movement.first("5 min")
            .last("4 min 30s")
            .cumulative_distance()
            .unwrap(["compute_track"])
            .filter(**self.filter_dict)  # type: ignore
            .diff("compute_track_unwrapped")
            .query(
                f"compute_track_unwrapped_diff.abs() > {self.track_threshold}"
            )
        )

        if direction_change is None:
            return None

        flight_before_direction_change = in_movement.before(
            direction_change.start
        )
        assert flight_before_direction_change is not None

        return flight_before_direction_change.assign(
            parking_position=parking_position.parking_position_max
        )


class ParkingAreaBasedPushback:
    """
    Detects and calculates pushback and taxiing characteristics for a given aircraft trajectory during takeoff.

    This function determines whether an aircraft has performed a pushback before taxiing by checking if the
    trajectory intersects defined stand areas. If pushback is detected, it calculates the start and duration of
    pushback, the start time of taxiing, as well as taxi duration and distance. It also updates the trajectory
    data with pushback and taxi attributes.

    Parameters:
    -----------
    traj : Trajectory
        The aircraft trajectory object containing time series data of the aircraft's ground movement and status.

    standAreas : list of Polygon
        List of polygonal areas representing possible stand locations where aircraft might initiate pushback.

    airport_str : str, optional
        String representing the airport code (default is 'LSZH') for location-specific data handling.

    Returns:
    --------
    traj : Trajectory
        Updated trajectory object with additional attributes:
        - isPushback (bool): Whether the aircraft performed a pushback.
        - startPushback (datetime): The start time of the pushback maneuver, if detected.
        - startTaxi (datetime): The start time of the taxiing phase.
        - pushbackDuration (timedelta): Duration of the pushback phase.
        - taxiDuration (timedelta): Duration of the taxiing phase.
        - taxiDistance (float): Total distance covered during taxiing in meters.

    Notes:
    ------
    - The function is specifically designed for analyzing takeoff events.
    - If the ground coverage is incomplete or trajectory segments are missing,
      adjustments may be made to the pushback and taxi timings.
    - The calculated `taxiDuration` and `taxiDistance` are only valid for takeoff movements
      with available lineup times.

    Example:
    --------
    traj = alternative_pushback_detection(traj, standAreas, airport_str='LSZH')
    """

    def __init__(
        self,
        airport: Union[str, "Airport"],
        stand_areas: Union[
            None, "Overpass", gpd.GeoDataFrame, list[Polygon]
        ] = None,
    ) -> None:
        from ...data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if self.airport is None:
            raise RuntimeError("Airport information missing")

        self.stand_areas: BaseGeometry

        if stand_areas is None:
            self.stand_areas = self.airport.apron.data.union_all()
        elif isinstance(stand_areas, list):
            self.stand_areas = unary_union(stand_areas)
        elif isinstance(stand_areas, gpd.GeoDataFrame):
            self.stand_areas = stand_areas.union_all()
        else:
            self.stand_areas = stand_areas.data.union_all()

    def apply(self, flight: Flight) -> Optional[Flight]:
        moving = (
            flight.cumulative_distance()
            .filter(compute_gs=3)
            .query("compute_gs > 1")
        )

        # Check that the aircraft is moving, at all
        if moving is None:
            _log.debug("not moving")
            return None

        median_filter = FilterMedian(compute_track_unwrapped=21, compute_gs=21)
        moving_it = (
            moving.unwrap(["compute_track"])
            .filter(filter=median_filter, strategy=None)
            .split("1 min")
        )

        # that's a first candidate for the pushback leg
        first = moving_it.next()
        assert first is not None

        # Check that the first movement intersects one of the stand areas
        if not first.intersects(self.stand_areas):
            _log.debug("not intersecting")
            return None

        # This is the piece of trajectory after pushback
        second = moving_it.next()
        # TODO check that we are still on the airport here...
        if second is None:
            _log.debug("no second")
            return None

        # We skip here few seconds so that movement is clearly initiated
        # before we capture the track angle
        second_skip = second.skip("5s")
        if second_skip is None:
            _log.debug("no second skip")
            return None

        # Compare computed track angles before and after the aircraft stops
        pushback_track = first.last("20s").compute_track_median
        taxi_track = second_skip.first("20s").compute_track_median

        # In theory, the difference should be 180°
        # but we are more flexible than that
        if abs(pushback_track - taxi_track) < 90:
            _log.debug("no angle")
            return None

        return flight.between(first.start, first.stop)
