from impunity import impunity

from ...core import Flight
from ...core import types as tt


class FlightPhasesOpenAP:
    """Assign a flight phase to each timestamp of a flight

    This implementation uses the OpenAP phase detection fuzzy logic method.

    Usage:

    >>> from traffic.data.samples import belevingsvlucht
    >>> climb = belevingsvlucht.phases().query('phase == "CLIMB"')
    >>> first_climb = climb.next("split")

    See also: :ref:`How to find flight phases on a trajectory?`

    We can confirm the non-empty intersection between the climb and take-off
    phases. The intersection operator ``&`` returns the part of the trajectory
    which is simultaneously a take-off and a climb:

    >>> takeoff = belevingsvlucht.next('takeoff("EHAM")')
    >>> (takeoff & first_climb).duration
    Timedelta('0 days 00:00:23')
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
