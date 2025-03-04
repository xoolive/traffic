from operator import attrgetter
from typing import Iterator

import pitot.geodesy as geo

import numpy as np
from shapely.geometry import Polygon

from ...core.distance import minimal_angular_difference
from ...core.flight import Flight
from ...core.structure import Airport


class PolygonBasedRunwayDetection:
    """Identifies the take-off runway for trajectories.

    Iterates on all segments of trajectory matching a zone around a runway
    of the  given airport. The takeoff runway number is appended as a new
    ``runway`` column.

    :param airport: The airport from where the flight takes off.
    :param max_ft_above_airport: maximum altitude AGL, relative to the
      airport, that a flight can be to be considered as aligned.
    :param zone_length: the length of the trapeze, aligned with the runway.
    :param little_base: the smallest base of the trapeze, on the runway
      threshold side.
    :param opening: the angle (in degrees) of opening of the trapeze.

    >>> from traffic.data.samples import belevingsvlucht
    >>> takeoff = belevingsvlucht.next('takeoff("EHAM", method="default")')
    >>> takeoff.duration
    Timedelta('0 days 00:00:25')
    """

    def __init__(
        self,
        airport: str | Airport,
        max_ft_above_airport: float = 2000,
        zone_length: int = 6000,
        little_base: int = 50,
        opening: float = 5,
    ):
        from ...data import airports

        self.max_ft_above_airport = max_ft_above_airport
        self.zone_length = zone_length
        self.little_base = little_base
        self.opening = opening

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if (
            self.airport is None
            or self.airport.runways is None
            or self.airport.runways.shape.is_empty
        ):
            raise RuntimeError("Airport or runway information missing")

        nb_run = len(self.airport.runways.data)
        self.alt = self.airport.altitude + self.max_ft_above_airport
        base = (
            self.zone_length * np.tan(self.opening * np.pi / 180)
            + self.little_base
        )

        # Create shapes around each runway
        list_p0 = geo.destination(
            list(self.airport.runways.data.latitude),
            list(self.airport.runways.data.longitude),
            list(self.airport.runways.data.bearing),
            [self.zone_length for i in range(nb_run)],
        )
        list_p1 = geo.destination(
            list(self.airport.runways.data.latitude),
            list(self.airport.runways.data.longitude),
            [x + 90 for x in list(self.airport.runways.data.bearing)],
            [self.little_base for i in range(nb_run)],
        )
        list_p2 = geo.destination(
            list(self.airport.runways.data.latitude),
            list(self.airport.runways.data.longitude),
            [x - 90 for x in list(self.airport.runways.data.bearing)],
            [self.little_base for i in range(nb_run)],
        )
        list_p3 = geo.destination(
            list_p0[0],
            list_p0[1],
            [x - 90 for x in list(self.airport.runways.data.bearing)],
            [base for i in range(nb_run)],
        )
        list_p4 = geo.destination(
            list_p0[0],
            list_p0[1],
            [x + 90 for x in list(self.airport.runways.data.bearing)],
            [base for i in range(nb_run)],
        )

        self.runway_polygons = {}

        for i, name in enumerate(self.airport.runways.data.name):
            lat = [list_p1[0][i], list_p2[0][i], list_p3[0][i], list_p4[0][i]]
            lon = [list_p1[1][i], list_p2[1][i], list_p3[1][i], list_p4[1][i]]

            poly = Polygon(zip(lon, lat))
            self.runway_polygons[name] = poly

    def apply(self, flight: Flight) -> Iterator[Flight]:
        low_traj = flight.phases().query(
            f"(phase == 'CLIMB' or phase == 'LEVEL') and altitude < {self.alt}"
        )

        if low_traj is None:
            return

        for segment in low_traj.split("2 min"):
            candidates_set = []
            for name, polygon in self.runway_polygons.items():
                if segment.intersects(polygon):
                    candidate = (
                        segment.cumulative_distance()
                        .clip_iterate(polygon)
                        .max(key="compute_gs_max")
                    )
                    if candidate is None or candidate.shape is None:
                        continue
                    start_runway = candidate.aligned(
                        self.airport, method="runway"
                    ).max()

                    if start_runway is not None:
                        candidate = candidate.after(start_runway.start)
                        if candidate is None or candidate.shape is None:
                            continue
                    if candidate.max("compute_gs") < 140:
                        continue

                    candidates_set.append(candidate.assign(runway=name))

            result = max(
                candidates_set, key=attrgetter("duration"), default=None
            )
            if result is not None:
                yield result


class TrackBasedRunwayDetection:
    """
    Determines the taking-off runway of a flight based on its surface trajectory

    :param airport: The airport from where the flight takes off.
    :param max_ft_above_airport: maximum altitude AGL, relative to the
      airport, that a flight can be to be considered as aligned.
    :param min_groundspeed_kts: Minimum groundspeed to consider.
    :param min_vert_rate_ftmin: Minimum vertical rate to consider.
    :param maximum_bearing_deg: Maximum bearing difference to consider.
    :param max_dist_nm: Maximum distance from the airport to consider.

    >>> from traffic.data.samples import elal747
    >>> takeoff = elal747.takeoff(method="track_based", airport="LIRF").next()
    >>> takeoff.duration, takeoff.runway_max
    (Timedelta('0 days 00:00:40'), '25')

    >>> from traffic.data.samples import belevingsvlucht
    >>> takeoff = belevingsvlucht.next('takeoff("EHAM", method="default")')
    >>> takeoff.duration
    Timedelta('0 days 00:00:25')

    """

    def __init__(
        self,
        airport: str | Airport,
        max_ft_above_airport: float = 1500,
        min_groundspeed_kts: float = 30,
        min_vert_rate_ftmin: float = 257,
        max_bearing_deg: float = 10,
        max_dist_nm: float = 5,
    ):
        from traffic.data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        self.max_ft_above_airport = max_ft_above_airport
        self.min_groundspeed_kts = min_groundspeed_kts
        self.min_vert_rate_ftmin = min_vert_rate_ftmin
        self.max_bearing_deg = max_bearing_deg
        self.max_dist_nm = max_dist_nm

    def apply(self, flight: "Flight") -> Iterator["Flight"]:
        # if not self.takeoff_from(self.airport):
        #     return None
        alt_max = self.airport.altitude + self.max_ft_above_airport
        if self.airport.runways is None:
            return None
        runways = self.airport.runways.data
        runways_names = runways.name

        filtered_flight = flight.distance(self.airport).query(
            f"distance < {self.max_dist_nm}"
        )
        if filtered_flight is None or filtered_flight.data.empty:
            return None
        if "geoaltitude" in filtered_flight.data.columns:
            query_str = (
                f"geoaltitude < {alt_max} and "
                f"vertical_rate > {self.min_vert_rate_ftmin} and "
                f"groundspeed > {self.min_groundspeed_kts}"
            )
        else:
            query_str = (
                f"altitude < {alt_max} and "
                f"vertical_rate > {self.min_vert_rate_ftmin} and "
                f"groundspeed > {self.min_groundspeed_kts}"
            )
        filtered_flight = filtered_flight.query(query_str)
        if (
            filtered_flight is None
            or filtered_flight.data.empty
            or len(filtered_flight.data) < 4
        ):
            return None

        # Check for parallel runways with suffixes L, R, or C
        has_parallel_runway = runways_names.str.contains(r"[LRC]").any()
        runway_bearings = runways.bearing

        median_track = filtered_flight.data["track"].median()
        closest_runway: None | str = None

        if not has_parallel_runway:
            # Find the runway with the bearing closest to the median track
            bearing_diff = runway_bearings.apply(
                lambda x: minimal_angular_difference(x, median_track)
            )
            closest_index = bearing_diff.idxmin()
            closest_runway = runways.name.iloc[closest_index]
        else:
            # Round runway bearings to the nearest 5 degrees
            rounded_bearings = (runway_bearings / 5).round() * 5
            rounded_bearings = (
                rounded_bearings % 360
            )  # Ensure bearings stay within 0-359

            # Find the bearing closest to the median track
            bearing_diff = rounded_bearings.apply(
                lambda x: minimal_angular_difference(x, median_track)
            )

            # Identify all runways where bearing diff is less than 10 deg

            candidate_runways = runways.loc[
                bearing_diff[bearing_diff < self.max_bearing_deg].index
            ]
            if candidate_runways.empty:
                return None
            elif len(candidate_runways) == 1:
                closest_runway = candidate_runways.name.iloc[0]
            else:
                # Calculate distance from flight trajectory
                # to each candidate runway
                flight_ls = filtered_flight.linestring
                rways_ls = self.airport.runways.shape

                closest = None
                for rway in rways_ls.geoms:
                    if closest is None:
                        closest = rway
                        continue
                    if rway.distance(flight_ls) < closest.distance(flight_ls):
                        closest = rway
                if closest is None:
                    return None
                lon_1, _, lon_2, _ = closest.bounds
                # candidate_runways =
                # lat/lon that is closest to the closest bounds
                eps = 1 / 1000
                closest_runway = candidate_runways.query(
                    f"(abs(longitude-{lon_1})<{eps}) or "
                    f"(abs(longitude-{lon_2})<{eps})"
                )["name"].iloc[0]
        yield filtered_flight.assign(runway=closest_runway)
