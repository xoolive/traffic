from datetime import timedelta
from typing import Dict, List, Union, cast

from pitot.geodesy import distance

import pandas as pd

from ...core.flight import Flight, Position
from ...core.flightplan import FlightPlan, _Point


class FlightPlanPredict:
    """
    A class to predict the flight path of an aircraft based on a given
    flight plan and prediction start time.

    The prediction is based on a set of navigational points (navaids), the
    aircraft's initial state at a specific start time and the last 3 minutes
    before the start of the prediction.

    Attributes:
        fp: The flight plan associated with the flight prediction.
        start (Union[str, pd.Timestamp]): The start time for the prediction.
        horizon_minutes (int): The time duration (in minutes) for the flight
            prediction.
        angle_precision (int): The precision for angle calculations when
            aligning on navigational points.
        min_distance (int): The minimum distance for considering
            navigational points during alignment.
    """

    def __init__(
        self,
        fp: FlightPlan | List[_Point],
        start: Union[str, pd.Timestamp],
        horizon_minutes: int = 15,
        angle_precision: int = 2,
        min_distance: int = 200,
        resample: str | None = "1s",
    ):
        self.fp = fp
        self.start = start
        self.horizon_minutes = horizon_minutes
        self.angle_precision = angle_precision
        self.min_distance = min_distance
        self.resample = resample

    # resample: None | str dans __init__
    def predict(self, flight: Flight) -> Flight:
        """
        Predict the flight path based on the provided flight and flight plan
        starting from a specific timestamp.

        This method computes the predicted flight path by aligning the flight
        with the given flight plan's navigational points. It considers the
        current position of the flight and predicts subsequent positions
        based on the flight plan and the aircraft's groundspeed during the
        previous 3 minutes.

        Args:
            flight: The flight object that contains the actual flight
            data.
            resample: Frequency at which to resample the flight data.


        Returns:
            Flight: A new Flight object corresponding to the prediction.

        Raises:
            AssertionError: If the flight or window data is invalid.
        """
        data_points: Dict[str, List[Union[float, str, pd.Timestamp]]] = {
            "latitude": [],
            "longitude": [],
            "timestamp": [],
            "groundspeed": [],
        }

        before_start = flight.before(self.start, strict=False)
        if before_start is None:
            raise RuntimeError("Flight should start before the start date")

        window = before_start.last(minutes=3)

        gs = window.groundspeed_mean * 0.514444  # m/s
        start_pos = before_start.at_ratio(1)

        assert start_pos is not None
        data_points["latitude"].append(start_pos.latitude)
        data_points["longitude"].append(start_pos.longitude)
        data_points["timestamp"].append(start_pos.timestamp)
        data_points["groundspeed"].append(gs / 0.514444)
        if isinstance(self.fp, FlightPlan):
            navaids = self.fp.all_points
        else:
            navaids = self.fp
        g = window.aligned_on_navpoint(
            self.fp,
            angle_precision=self.angle_precision,
            min_distance=self.min_distance,
        ).final()
        start_nav_name = cast(Flight, g).data.navaid.iloc[0]
        start_nav = next(
            (point for point in navaids if point.name == start_nav_name), None
        )
        start_index = navaids.index(cast(_Point, start_nav))
        rest_navaids = navaids[start_index:]

        point_depart = _Point(
            lat=start_pos.latitude,
            lon=start_pos.longitude,
            name=start_nav_name,
        )
        new_timestamp = pd.Timestamp(self.start)

        for navaid in rest_navaids:
            dmin = distance(
                point_depart.latitude,
                point_depart.longitude,
                navaid.latitude,
                navaid.longitude,
            )
            t = int(dmin / gs)
            new_timestamp = new_timestamp + timedelta(seconds=t)
            point_depart = navaid

            data_points["latitude"].append(navaid.latitude)
            data_points["longitude"].append(navaid.longitude)
            data_points["timestamp"].append(new_timestamp)
            data_points["groundspeed"].append(gs / 0.514444)

            tdiff_minutes = (new_timestamp - self.start).total_seconds() / 60
            if (
                tdiff_minutes > self.horizon_minutes
                and len(data_points["timestamp"]) > 1
            ):
                break

        new_columns = {
            **data_points,
            "icao24": flight.icao24,
            "callsign": flight.callsign,
            "altitude": cast(Position, flight.at(self.start)).altitude,
            "flight_id": flight.flight_id,
        }
        ret = Flight(pd.DataFrame(new_columns))

        if self.resample is not None:
            ret = ret.resample(self.resample).first(
                self.horizon_minutes * 60 + 1
            )

        ret = ret.cumulative_distance(compute_track=True).rename(
            columns={"compute_track": "track"}
        )

        return ret
