from operator import attrgetter
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np
import pandas as pd

from ...core.flight import Flight
from ...core.structure import Airport
from ...core.time import deltalike

if TYPE_CHECKING:
    from ...data.basic.airports import Airports


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


    >>> from traffic.data.samples import belevingsvlucht

    The detected runway identifier is available in the ILS column:


    >>> aligned = belevingsvlucht.landing('EHAM').next()
    >>> f"ILS {aligned.max('ILS')}; landing time {aligned.stop:%H:%M}"
    'ILS 06; landing time 20:17'

    We can specify the method by default:


    >>> for aligned in belevingsvlucht.landing('EHLE', method="default"):
    ...     print(aligned.start)
    2018-05-30 16:00:55+00:00
    2018-05-30 16:50:44+00:00
    2018-05-30 17:21:17+00:00
    2018-05-30 18:13:02+00:00
    2018-05-30 19:05:22+00:00
    2018-05-30 19:42:36+00:00

    >>> final = belevingsvlucht.final("landing('EHLE', method='default')")
    >>> final.ILS_max  # equivalent to final.max("ILS")
    '23'

    Usual built-in functions works as on any kind of iterators, here we get the
    flight segment with the latest start timestamp. (equivalent to the `final`
    method used above)

    >>> from operator import attrgetter
    >>> last_aligned = max(
    ...     belevingsvlucht.landing("EHLE"),
    ...     key=attrgetter('start')
    ... )
    >>> last_aligned.start
    Timestamp('2018-05-30 19:42:36+0000', tz='UTC')


    """

    def __init__(
        self,
        airport: str | Airport,
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
    """Detects runway changes.

    The method yields pieces of trajectories with exactly two runway alignments
    on the same airport not separated by a climbing phase.

    In each piece of yielded trajectory, the ``ILS`` column contains the name of
    the runway targetted by the aircraft at each instant.

    :param airport: Airport where the ILS is located
    :param kwargs: are passed to the constructor of :class:`LandingAlignedOnILS`

    We can reuse the trajectory from the :ref:`Getting started` page to
    illustrate the usage of this method

    >>> from traffic.data.samples import quickstart
    >>> flight = quickstart['AFR17YC']
    >>> flight.has("landing(airport='LFPG', method='runway_change')")
    True

    Since the trajectory has a runway change, we can print characteristics of
    each segment:

    >>> segments = flight.landing('LFPG', method="aligned_on_ils")
    >>> first = next(segments)
    >>> f"{first.start:%H:%M} {first.stop:%H:%M} aligned on {first.ILS_max}"
    '13:34 13:36 aligned on 08R'
    >>> second = next(segments)
    >>> f"{second.start:%H:%M} {second.stop:%H:%M} aligned on {second.ILS_max}"
    '13:36 13:39 aligned on 08L'
    """

    def __init__(
        self,
        airport: str | Airport,
        **kwargs: Any,
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

        self.aligned_on_ils = LandingAlignedOnILS(airport=airport, **kwargs)

    def apply(self, flight: Flight) -> Iterator[Flight]:
        aligned = iter(flight.landing(self.aligned_on_ils))
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
                    airport=self.airport
                    if isinstance(self.airport, str)
                    else self.airport.icao
                )

            first = second


class LandingAnyAttempt:
    """Iterates on landing attempts without specific airport information.

    First, candidates airports are identified in the neighbourhood of the
    segments of trajectory below a threshold altitude. By default, the full
    airport database is considered but it is possible to restrict it and pass a
    smaller database with the dataset parameter.

    If no runway information is available for the given airport, no
    trajectory segment will be provided.

    :param dataset: a subset of airports used as candidate landing airports
    :param alt_threshold: an altitude threshold helpful to select candidate
      airports. It is important to pick a threshold significantly above the
      altitude of the candidate airports.

    >>> from traffic.data.samples import belevingsvlucht
    >>> attempts = belevingsvlucht.landing(method="any")
    >>> for i, attempt in enumerate(attempts):
    ...     print(f"Step {i}: {attempt.airport_max} runway {attempt.ILS_max}")
    Step 0: EHLE runway 23
    Step 1: EHLE runway 05
    Step 2: EHLE runway 23
    Step 3: EHLE runway 05
    Step 4: EHLE runway 23
    Step 5: EHAM runway 06

    """

    def __init__(
        self,
        dataset: Optional["Airports"] = None,
        alt_threshold: float = 8000,
        **kwargs: Any,
    ):
        self.dataset = dataset
        self.alt_threshold = alt_threshold
        self.kwargs = kwargs

    def apply(self, flight: Flight) -> Iterator[Flight]:
        from ..metadata.airports import LandingAirportInference

        candidate = flight.query(f"altitude < {self.alt_threshold}")
        if candidate is not None:
            for chunk in candidate.split("10 min"):
                point = chunk.query("altitude == altitude.min()")
                if point is None:
                    return
                candidate_airport = point.infer_airport(
                    method=LandingAirportInference(dataset=self.dataset)
                )
                if candidate_airport.runways is not None:
                    chunk = chunk.assign(airport=candidate_airport.icao)
                    method = LandingAlignedOnILS(
                        airport=candidate_airport, **self.kwargs
                    )
                    yield from chunk.landing(method=method)
