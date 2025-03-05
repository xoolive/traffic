from datetime import timedelta
from typing import Any

from impunity import impunity
from pitot import geodesy as geo

import pandas as pd

from ...core import types as tt
from ...core.flight import Flight


class StraightLinePredict:
    """Projects the trajectory in a straight line.

    The method uses the last position of a trajectory (method
    :meth:`~traffic.core.Flight.at`) and uses the ``track`` (in degrees),
    ``groundspeed`` (in knots) and ``vertical_rate`` (in ft/min) values to
    interpolate the trajectory in a straight line.

    The elements passed as kwargs as passed as is to the datetime.timedelta
    constructor.
    """

    def __init__(
        self, forward: None | str | pd.Timedelta = None, **kwargs: Any
    ):
        if isinstance(forward, str):
            delta = pd.Timedelta(forward)
        elif forward is None:
            delta = timedelta(**kwargs)
        else:
            delta = forward

        self.forward = delta

    @impunity(ignore_warnings=True)
    def predict(self, flight: Flight) -> Flight:
        last_line = flight.at()
        if last_line is None:
            raise ValueError("Unknown data for this flight")
        window = flight.last(seconds=20)

        if window is None:
            raise RuntimeError("Flight expect at least 20 seconds of data")

        new_gs: tt.speed = window.data.groundspeed.mean()
        new_vr: tt.vertical_rate = window.data.vertical_rate.mean()
        duration: tt.seconds = self.forward.total_seconds()

        new_lat, new_lon, _ = geo.destination(
            last_line.latitude,
            last_line.longitude,
            last_line.track,
            new_gs * duration,
        )

        last_alt: tt.altitude = last_line.altitude
        new_alt: tt.altitude = last_alt + new_vr * duration

        return Flight(
            pd.DataFrame.from_records(
                [
                    last_line,
                    pd.Series(
                        {
                            "timestamp": last_line.timestamp + self.forward,
                            "latitude": new_lat,
                            "longitude": new_lon,
                            "altitude": new_alt,
                            "groundspeed": new_gs,
                            "vertical_rate": new_vr,
                        }
                    ),
                ]
            ).ffill()
        )
