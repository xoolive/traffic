from typing import TYPE_CHECKING, Iterator, Optional

from ...core import Flight
from ...core.structure import Airport

if TYPE_CHECKING:
    from cartes.osm import Overpass


class ParkingPositionGeometricIntersection:
    """
    Generates possible parking positions at a given airport.

    Example usage:

    parking = flight.on_parking_position('LSZH').max()
    # returns the most probable parking position in terms of duration

    .. warning::

        This method has been well tested for aircraft taking off, but should
        be double checked for landing trajectories.

    """

    def __init__(
        self,
        airport: str | Airport,
        buffer_size: float = 1e-5,  # degrees
        parking_positions: Optional["Overpass"] = None,
    ) -> None:
        from ...data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if self.airport is None:
            raise RuntimeError("Airport information missing")

        self.buffer_size = buffer_size
        self.parking_positions = parking_positions

    def apply(self, flight: "Flight") -> Iterator["Flight"]:
        inside_airport = flight.filter().inside_bbox(self.airport)
        if inside_airport is None:
            return None

        inside_airport = inside_airport.split().max()

        parking_positions = (
            self.airport.parking_position.query("type_ == 'way'")
            if self.parking_positions is None
            else self.parking_positions.query("type_ == 'way'")
        )
        for _, p in parking_positions.data.iterrows():
            if inside_airport.intersects(p.geometry.buffer(self.buffer_size)):
                parking_part = inside_airport.clip(
                    p.geometry.buffer(self.buffer_size)
                )
                if parking_part is not None:
                    yield parking_part.assign(parking_position=p.ref)
