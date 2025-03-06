import logging
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ...core.flight import Flight
from ...core.structure import Airport
from ..filters import FilterMedian

if TYPE_CHECKING:
    from cartes.osm import Overpass

    from shapely.geometry import Polygon

_log = logging.getLogger(__name__)


class ParkingPositionBasedPushback:
    """
    Returns the pushback part of the trajectory on ground.

    The method identifies the start of the movement, the parking_position
    and the moment the aircraft suddenly changes direction in the computed
    track angle.

    :param airport: Airport where the ILS is located
    :param filter_dict:
    :param track_threshold:
    :param parking_positions: The parking positions can be passed as an
      :class:`~cartes.osm.Overpass` instance.

    >>> from traffic.data.samples import zurich_airport
    >>> flight = zurich_airport["AEE5ZH"]
    >>> pushback = flight.pushback('LSZH', method="parking_position")
    >>> pushback.duration
    Timedelta('0 days 00:01:45')

    .. warning::

        The method has poor performance when trajectory point on ground are
        lacking. This is often the case for data recorded from locations far
        from the airport.

    """

    def __init__(
        self,
        airport: Union[str, Airport],
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

    def apply(self, flight: Flight) -> Optional[Flight]:
        within_airport: Optional[Flight] = flight.inside_bbox(self.airport)
        if within_airport is None:
            return None

        parking_position = within_airport.parking_position(
            self.airport, parking_positions=self.parking_positions
        ).next()
        if parking_position is None:
            return None

        after_parking = within_airport.after(parking_position.start)
        assert after_parking is not None

        in_movement = after_parking.movement(method="start_moving")

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
    Returns the pushback part of the trajectory on ground.

    The method identifies the start of the movement, an intersection with a
    documented apron area and the moment the aircraft suddenly changes direction
    in the computed track angle.

    :param airport: Airport where the ILS is located
    :param stand_areas: The parking positions can be passed as an
      :class:`~cartes.osm.Overpass` instance or a list of Polygon.


    >>> from traffic.data.samples import zurich_airport
    >>> flight = zurich_airport["AEE5ZH"]
    >>> pushback = flight.pushback('LSZH', method="parking_area")
    >>> pushback.duration
    Timedelta('0 days 00:04:26')

    .. warning::

        The method has poor performance when trajectory point on ground are
        lacking. This is often the case for data recorded from locations far
        from the airport.

    """

    def __init__(
        self,
        airport: str | Airport,
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
