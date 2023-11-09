from datetime import timedelta
from traffic.core.flightplan import _Point
from traffic.core.geodesy import distance  # distance in meters
from traffic.core.traffic import Flight
from traffic.core import FlightPlan
import pandas as pd
from traffic.data import aixm_navaids


def predict_fp(
    flight: Flight,
    fp: FlightPlan,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    minutes: str = 15,  # REPLACE WITH INT
    angle_precision: int = 2,
    min_distance: int = 150,
):
    data_points = {"latitude": [], "longitude": [], "timestamp": []}
    # print(f"start = {start}\nstop = {stop}\n")
    assert flight is not None
    trou = flight.between(start, stop, strict=False)
    assert trou is not None
    section = flight.before(trou.start, strict=False).last(minutes=20)
    assert section is not None
    gs = section.groundspeed_mean * 0.514444  # conversion to m/s

    data_points["latitude"].append(
        flight.before(trou.start).at_ratio(1).latitude
    )
    data_points["longitude"].append(
        flight.before(trou.start).at_ratio(1).longitude
    )
    data_points["timestamp"].append(
        flight.before(trou.start).at_ratio(1).timestamp
    )

    navaids = fp.all_points

    # initialize first navaid
    g = section.aligned_on_navpoint(
        fp, angle_precision=angle_precision, min_distance=min_distance
    ).final()

    start_nav_name = g.data.navaid.iloc[0]
    start_nav = next(
        (point for point in navaids if point.name == start_nav_name), None
    )
    start_index = navaids.index(start_nav)
    reste_navaids = navaids[start_index:]
    point_depart = _Point(
        lat=flight.before(trou.start).at_ratio(1).latitude,
        lon=flight.before(trou.start).at_ratio(1).longitude,
        name=start_nav_name,
    )
    new_timestamp = trou.start
    for navaid in reste_navaids:
        dmin = distance(
            point_depart.latitude,
            point_depart.longitude,
            navaid.latitude,
            navaid.longitude,
        )
        t = int(dmin / gs)
        # new_timestamp = trou.start + timedelta(seconds=t)
        new_timestamp = new_timestamp + timedelta(seconds=t)
        point_depart = navaid
        data_points["latitude"].append(navaid.latitude)
        data_points["longitude"].append(navaid.longitude)
        data_points["timestamp"].append(new_timestamp)
        # compute difference between trou.start and new_timestamp
        time_difference_seconds = (new_timestamp - trou.start).total_seconds()
        time_difference_minutes = time_difference_seconds / 60
        if (
            time_difference_minutes > minutes
            and len(data_points["timestamp"]) > 1
        ):
            break

    new_columns = {
        **data_points,
        "icao24": flight.icao24,
        "callsign": flight.callsign,
        "altitude": flight.at(trou.start).altitude,
        "flight_id": flight.flight_id,
    }
    return (
        Flight(pd.DataFrame(new_columns)).resample("1s").first(minutes * 60 + 1)
    )
