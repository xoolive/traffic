import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union, cast

from impunity import impunity
from pitot.geodesy import distance

import pandas as pd
from traffic.core.flight import Flight, Position
from traffic.core.flightplan import FlightPlan, _Point


def angle_from_coordinates(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing angle between two geographical coordinates.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: The bearing angle in degrees between the two points, ranging from 0 to 360.
    """
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dLon
    )
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    return (brng + 360) % 360


class FlightPlanPredict:
    """
    A class to predict the flight path of an aircraft based on a given
    flight plan and prediction start time.

    The prediction is based on a set of navigational points (navaids), the
    aircraft's initial state at a specific start time and the last 3 minutes
    before the start of the prediction.

    Attributes:
        fp (FlightPlan): The flight plan associated with the flight prediction.
        start (Union[str, pd.Timestamp]): The start time for the prediction.
        minutes (int): The time duration (in minutes) for the flight prediction.
        angle_precision (int): The precision for angle calculations when
        aligning on navigational points.
        min_distance (int): The minimum distance for considering
        navigational points during alignment.
    """

    def __init__(
        self,
        fp: FlightPlan,
        start: Union[str, pd.Timestamp],
        minutes: int = 15,
        angle_precision: int = 2,
        min_distance: int = 200,
    ):
        self.fp = fp
        self.start = start
        self.minutes = minutes
        self.angle_precision = angle_precision
        self.min_distance = min_distance

    def predict(self, flight: Flight, resample: bool = True) -> Flight:
        """
        Predict the flight path based on the provided flight and flight plan
        starting from a specific timestamp.

        This method computes the predicted flight path by aligning the flight
        with the given flight plan's navigational points. It considers the
        current position of the flight and predicts subsequent positions
        based on the flight plan and the aircraft's groundspeed during the
        previous 3 minutes.

        Args:
            flight (Flight): The flight object that contains the actual flight
            data.
            resample (bool, optional): Whether to resample the flight data at
            1-second intervals. Default is True.

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

        assert flight is not None
        window = flight.before(self.start, strict=False).last(minutes=3)
        assert window is not None

        gs = window.groundspeed_mean * 0.514444  # m/s

        start_pos = cast(Position, flight.before(self.start).at_ratio(1))
        data_points["latitude"].append(start_pos.latitude)
        data_points["longitude"].append(start_pos.longitude)
        data_points["timestamp"].append(start_pos.timestamp)
        data_points["groundspeed"].append(gs / 0.514444)

        navaids = self.fp.all_points
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
            lat=start_pos.latitude, lon=start_pos.longitude, name=start_nav_name
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

            if (new_timestamp - self.start).total_seconds() / 60 > self.minutes and len(
                data_points["timestamp"]
            ) > 1:
                break

        new_columns = {
            **data_points,
            "icao24": flight.icao24,
            "callsign": flight.callsign,
            "altitude": cast(Position, flight.at(self.start)).altitude,
            "flight_id": flight.flight_id,
        }
        ret = Flight(pd.DataFrame(new_columns))

        if resample:
            ret = ret.resample("1s").first(self.minutes * 60 + 1)

        ret = ret.cumulative_distance(compute_track=True).rename(
            columns={"compute_track": "track"}
        )

        return ret
