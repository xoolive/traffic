from operator import attrgetter
from typing import TYPE_CHECKING, Iterator, Protocol, Union

import pitot.geodesy as geo

import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    from ...core.flight import Flight
    from ...core.structure import Airport


class Takeoff(Protocol):
    def apply(self, flight: "Flight") -> Iterator["Flight"]: ...


class Default(Takeoff):
    """Identifies the take-off runway for trajectories.

    Iterates on all segments of trajectory matching a zone around a runway
    of the  given airport. The takeoff runway number is appended as a new
    ``runway`` column.

    """

    def __init__(
        self,
        airport: Union[str, "Airport"],
        threshold_alt: int = 2000,
        zone_length: int = 6000,
        little_base: int = 50,
        opening: float = 5,
    ):
        self.airport = airport
        self.threshold_alt = threshold_alt
        self.zone_length = zone_length
        self.little_base = little_base
        self.opening = opening

    def apply(self, flight: "Flight") -> Iterator["Flight"]:
        from ...data import airports

        flight = flight.phases()

        _airport = (
            airports[self.airport]
            if isinstance(self.airport, str)
            else self.airport
        )
        if (
            _airport is None
            or _airport.runways is None
            or _airport.runways.shape.is_empty
        ):
            return None

        nb_run = len(_airport.runways.data)
        alt = _airport.altitude + self.threshold_alt
        base = (
            self.zone_length * np.tan(self.opening * np.pi / 180)
            + self.little_base
        )

        # Create shapes around each runway
        list_p0 = geo.destination(
            list(_airport.runways.data.latitude),
            list(_airport.runways.data.longitude),
            list(_airport.runways.data.bearing),
            [self.zone_length for i in range(nb_run)],
        )
        list_p1 = geo.destination(
            list(_airport.runways.data.latitude),
            list(_airport.runways.data.longitude),
            [x + 90 for x in list(_airport.runways.data.bearing)],
            [self.little_base for i in range(nb_run)],
        )
        list_p2 = geo.destination(
            list(_airport.runways.data.latitude),
            list(_airport.runways.data.longitude),
            [x - 90 for x in list(_airport.runways.data.bearing)],
            [self.little_base for i in range(nb_run)],
        )
        list_p3 = geo.destination(
            list_p0[0],
            list_p0[1],
            [x - 90 for x in list(_airport.runways.data.bearing)],
            [base for i in range(nb_run)],
        )
        list_p4 = geo.destination(
            list_p0[0],
            list_p0[1],
            [x + 90 for x in list(_airport.runways.data.bearing)],
            [base for i in range(nb_run)],
        )

        runway_polygons = {}

        for i, name in enumerate(_airport.runways.data.name):
            lat = [list_p1[0][i], list_p2[0][i], list_p3[0][i], list_p4[0][i]]
            lon = [list_p1[1][i], list_p2[1][i], list_p3[1][i], list_p4[1][i]]

            poly = Polygon(zip(lon, lat))
            runway_polygons[name] = poly

        low_traj = flight.query(
            f"(phase == 'CLIMB' or phase == 'LEVEL') and altitude < {alt}"
        )

        if low_traj is None:
            return

        for segment in low_traj.split("2 min"):
            candidates_set = []
            for name, polygon in runway_polygons.items():
                if segment.intersects(polygon):
                    candidate = (
                        segment.cumulative_distance()
                        .clip_iterate(polygon)
                        .max(key="compute_gs_max")
                    )
                    if candidate is None or candidate.shape is None:
                        continue
                    start_runway = candidate.aligned_on_runway(_airport).max()

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
