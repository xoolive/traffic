from typing import TYPE_CHECKING, cast

import numpy as np
from openap.phase import FlightPhase

if TYPE_CHECKING:
    from ..core import Flight  # noqa: F401


class FuzzyLogic:
    def phases(self, twindow: int = 60) -> "Flight":
        """Assign a flight phase to each timestamp of a flight
        using OpenAP phase detection fuzzy logic method.
        """

        # The following cast secures the typing
        self = cast("Flight", self)

        fp = FlightPhase()
        fp.set_trajectory(
            (self.data.timestamp.values - np.datetime64("1970-01-01"))
            / np.timedelta64(1, "s"),
            self.data.altitude.values,
            self.data.groundspeed.values,
            self.data.vertical_rate.values,
        )
        return self.assign(phase=fp.phaselabel(twindow=twindow)).assign(
            phase=lambda df: df.phase.str.replace("GND", "GROUND")
            .str.replace("CL", "CLIMB")
            .str.replace("DE", "DESCENT")
            .str.replace("CR", "CRUISE")
            .str.replace("LVL", "LEVEL")
        )
