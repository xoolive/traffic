import logging
from typing import TYPE_CHECKING, Optional, Protocol

from ...core import Flight
from ...core.distance import closest_point
from ...core.mixins import PointMixin
from ...core.structure import Airport

if TYPE_CHECKING:
    from ...data.basic.airports import Airports

_log = logging.getLogger(__name__)


class AirportInferenceBase(Protocol):
    def infer(self, flight: Flight) -> Airport: ...


def guess_airport(
    point: None | PointMixin = None,
    *,
    latitude: None | float = None,
    longitude: None | float = None,
    dataset: Optional["Airports"] = None,
    warning_distance: None | float = None,
) -> Airport:
    if dataset is None:
        from ...data import airports

        dataset = airports

    # TODO define a protocol instead of PointMixin
    if point is not None:
        longitude = point.longitude
        latitude = point.latitude
    if any((longitude is None, latitude is None)):
        raise RuntimeError("latitude or longitude are None")

    airport_data = closest_point(
        dataset.data, latitude=latitude, longitude=longitude
    )
    airport = Airport(
        airport_data.get("altitude"),
        airport_data.get("country"),
        airport_data.get("iata"),
        airport_data.get("icao"),
        airport_data.get("latitude"),
        airport_data.get("longitude"),
        airport_data.get("name"),
    )
    airport.distance = airport_data.distance  # type: ignore

    if warning_distance is not None and airport.distance > warning_distance:
        _log.warning(
            f"Closest airport is more than {warning_distance * 1e-3}km away "
            f" (distance={airport.distance})"
        )
    return airport


class TakeoffAirportInference:
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

    def __init__(
        self,
        dataset: Optional["Airports"] = None,
        warning_distance: None | float = None,
    ):
        self.dataset = dataset
        self.warning_distance = warning_distance

    def infer(self, flight: Flight) -> Airport:
        data = flight.data.sort_values("timestamp")
        return guess_airport(
            data.iloc[0],
            dataset=self.dataset,
            warning_distance=self.warning_distance,
        )


class LandingAirportInference:
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

    def __init__(
        self,
        dataset: Optional["Airports"] = None,
        warning_distance: None | float = None,
    ):
        self.dataset = dataset
        self.warning_distance = warning_distance

    def infer(self, flight: Flight) -> Airport:
        data = flight.data.sort_values("timestamp")
        return guess_airport(
            data.iloc[-1],
            dataset=self.dataset,
            warning_distance=self.warning_distance,
        )
