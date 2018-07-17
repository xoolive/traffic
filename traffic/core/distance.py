import geodesy.sphere as geo

from typing import Optional, NamedTuple

import pandas as pd
# from ..data.basic.airport import Airport


class DistanceAirport(NamedTuple):
    distance: float
    # TODO I would like to specify a 'Point' trait in place of NamedTuple
    airport: NamedTuple


class DistancePointTrajectory(NamedTuple):
    distance: float
    name: str
    point: pd.Series


def closest_point(
    data: pd.DataFrame,
    point: Optional[NamedTuple] = None,
    *args,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> DistancePointTrajectory:

    if point is not None:
        latitude = point.latitude  # type: ignore
        longitude = point.longitude  # type: ignore
        name = point.name  # type: ignore
    else:
        name = "unnamed"
    dist_vect = geo.distance(
        data.latitude,
        data.longitude,
        [latitude for _ in data.latitude],
        [longitude for _ in data.longitude],
    )
    argmin = dist_vect.argmin()
    return DistancePointTrajectory(dist_vect[argmin], name, data.iloc[argmin])


def guess_airport(
    point: Optional[NamedTuple] = None,
    *args,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> DistanceAirport:
    from ..data import airports

    if point is not None:
        longitude = point.longitude  # type: ignore
        latitude = point.latitude  # type: ignore
    if any((longitude is None, latitude is None)):
        raise RuntimeError("latitude or longitude are None")

    airports_df = pd.DataFrame.from_records(
        a._asdict() for a in airports.airports
    ).rename(columns={"lat": "latitude", "lon": "longitude"})

    distance, _, airport = closest_point(
        airports_df, latitude=latitude, longitude=longitude
    )

    return DistanceAirport(distance, airports[airport.icao])
