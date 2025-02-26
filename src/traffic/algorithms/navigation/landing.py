from operator import attrgetter
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol, Union

import numpy as np
import pandas as pd

from ...core.flight import Flight
from ...core.structure import Airport
from ...core.time import deltalike

if TYPE_CHECKING:
    from ...data.basic.airports import Airports


class LandingBase(Protocol):
    def apply(self, flight: Flight) -> Iterator[Flight]: ...


class LandingAlignedOnILS:
    """Iterates on all segments of trajectory aligned with the ILS of the
    given airport. The runway number is appended as a new ``ILS`` column.

    :param airport: Airport where the ILS is located
    :param angle_tolerance: maximum tolerance on bearing difference between
        ILS and flight trajectory.
    :param min_duration: minimum duration a flight has to spend on the ILS
        to be considered as aligned.
    :param max_ft_above_airport: maximum altitude AGL, relative to the
        airport, that a flight can be to be considered as aligned.

    Example usage:

    .. code:: python

        >>> aligned = belevingsvlucht.landing('EHAM').next()
        >>> f"ILS {aligned.max('ILS')} until {aligned.stop:%H:%M}"
        'ILS 06 until 20:17'

    .. code:: python

        >>> for aligned in belevingsvlucht.landing('EHLE'):
        ...     print(aligned.start)
        2018-05-30 16:50:44+00:00
        2018-05-30 18:13:02+00:00
        2018-05-30 16:00:55+00:00
        2018-05-30 17:21:17+00:00
        2018-05-30 19:05:22+00:00
        2018-05-30 19:42:36+00:00

        >>> from operator import attrgetter
        >>> last_aligned = max(
        ...     belevingsvlucht.landing("EHLE"),
        ...     key=attrgetter('start')
        ... )
    """

    def __init__(
        self,
        airport: Union[str, Airport],
        angle_tolerance: float = 0.1,
        min_duration: deltalike = "1 min",
        max_ft_above_airport: float = 5000,
    ):
        from ...data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if (
            self.airport is None
            or self.airport.runways is None
            or self.airport.runways.shape.is_empty
        ):
            raise RuntimeError("Airport or runway information missing")

        self.angle_tolerance = angle_tolerance
        self.min_duration = min_duration
        self.max_ft_above_airport = max_ft_above_airport

    def apply(self, flight: Flight) -> Iterator[Flight]:
        rad = np.pi / 180

        chunks = list()
        for threshold in self.airport.runways.list:
            tentative = (
                flight.bearing(threshold)
                .distance(threshold)
                .assign(
                    b_diff=lambda df: df.distance
                    * np.radians(df.bearing - threshold.bearing).abs()
                )
                .query(
                    f"b_diff < {self.angle_tolerance} and "
                    f"cos((bearing - track) * {rad}) > 0"
                )
            )
            if tentative is not None:
                for chunk in tentative.split("20s"):
                    if (
                        chunk.longer_than(self.min_duration)
                        and not pd.isna(altmin := chunk.altitude_min)
                        and altmin
                        < (self.airport.altitude or 0)
                        + self.max_ft_above_airport
                    ):
                        chunks.append(
                            chunk.assign(
                                ILS=threshold.name, airport=self.airport.icao
                            )
                        )

        yield from sorted(chunks, key=attrgetter("start"))


class LandingWithRunwayChange:
    def apply(
        self,
        flight: Flight,
        airport: None | str | Airport = None,
        dataset: Optional["Airports"] = None,
        **kwargs: Any,
    ) -> Iterator[Flight]:
        """Detects runway changes.

        The method yields pieces of trajectories with exactly two runway
        alignments on the same airport not separated by a climbing phase.

        In each piece of yielded trajectory, the `ILS` column contains the
        name of the runway targetted by the aircraft at each instant.
        """

        if airport is None:
            if dataset is None:
                airport = flight.landing_airport()
            else:
                airport = flight.landing_airport(dataset=dataset)

        if airport is None:
            return None

        aligned = iter(flight.aligned_on_ils(airport, **kwargs))
        first = next(aligned, None)
        if first is None:
            return

        for second in aligned:
            candidate = flight.between(first.start, second.stop)
            assert candidate is not None
            candidate = candidate.assign(ILS=None)
            if candidate.phases().query('phase == "CLIMB"') is None:
                candidate.data.loc[
                    candidate.data.timestamp <= first.stop, "ILS"
                ] = first.max("ILS")
                candidate.data.loc[
                    candidate.data.timestamp >= second.start, "ILS"
                ] = second.max("ILS")

                yield candidate.assign(
                    airport=airport
                    if isinstance(airport, str)
                    else airport.icao
                )

            first = second
