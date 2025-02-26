from typing import Protocol

from impunity import impunity

from ...core import Flight
from ...core import types as tt


class FlightPhasesBase(Protocol):
    def apply(self, flight: Flight) -> Flight: ...


class FlightPhasesOpenAP:
    """Assign a flight phase to each timestamp of a flight
    using OpenAP phase detection fuzzy logic method.
    """

    def __init__(self, twindow: int = 60) -> None:
        self.twindow = twindow

    @impunity
    def apply(self, flight: Flight) -> Flight:
        from openap.phase import FlightPhase

        altitude: tt.altitude_array = flight.data.altitude.values
        groundspeed: tt.speed_array = flight.data.groundspeed.values
        vertical_rate: tt.vertical_rate_array = flight.data.vertical_rate.values

        fp = FlightPhase()
        fp.set_trajectory(
            flight.data.timestamp.dt.as_unit("s").astype(int).values,
            altitude,
            groundspeed,
            vertical_rate,
        )
        return flight.assign(phase=fp.phaselabel(twindow=self.twindow)).assign(
            phase=lambda df: df.phase.str.replace("GND", "GROUND")
            .str.replace("CL", "CLIMB")
            .str.replace("DE", "DESCENT")
            .str.replace("CR", "CRUISE")
            .str.replace("LVL", "LEVEL")
        )
