from typing import TYPE_CHECKING, Iterator, Optional

from ...core import Flight
from ...core.structure import Airport
from .landing import LandingAlignedOnILS

if TYPE_CHECKING:
    from ...data.basic.airports import Airports


class GoAroundDetection:
    """Detects go-arounds.

    The method yields pieces of trajectories with exactly two landing attempts
    (aligned on one runway) on the same airport separated by exactly one
    climbing phase.

    :param airport: If None, the method tries to guess the landing airport
        based on the ``dataset`` parameter. (see
        :meth:`~traffic.core.Flight.landing_airport`)

    :param dataset: database of candidate airports, only used if ``airport``
        is None

    **See also:** :ref:`How to select go-arounds from a set of
    trajectories?`

    Example usage:

    >>> from traffic.data.samples import belevingsvlucht

    By default, go arounds will be detected at Amsterdam Schiphol airport (EHAM)


    >>> belevingsvlucht.go_around().next()
    >>> amsterdam_goaround = GoAroundDetection(airport="EHAM")
    >>> belevingsvlucht.go_around(method=amsterdam_goaround).next()

    There were none; however we find 5 of them at Lelystad airport.

    >>> belevingsvlucht.go_around("EHLE").sum()
    5

    """

    def __init__(
        self,
        airport: None | str | Airport = None,
        dataset: Optional["Airports"] = None,
    ) -> None:
        self.airport = airport
        self.dataset = dataset
        self.landing_ils = (
            LandingAlignedOnILS(airport=airport)
            if airport is not None
            else None
        )

    def apply(self, flight: Flight) -> Iterator[Flight]:
        if self.airport is None:
            self.airport = flight.infer_airport("landing", dataset=self.dataset)

        if self.airport is None:
            return None

        landing_ils = (
            self.landing_ils
            if self.landing_ils is not None
            else LandingAlignedOnILS(self.airport)
        )

        attempts = flight.landing(method=landing_ils)
        # you need to be aligned at least twice on a rway to have a GA:
        if len(attempts) < 2:
            return

        first_attempt = next(attempts, None)

        while first_attempt is not None:
            after_first_attempt = flight.after(first_attempt.start)
            assert after_first_attempt is not None

            climb = after_first_attempt.phases().query('phase == "CLIMB"')
            if climb is None:
                return

            after_climb = flight.after(next(climb.split("10 min")).stop)
            if after_climb is None:
                return

            next_attempt = next(after_climb.landing(method=landing_ils), None)

            if next_attempt is not None:
                goaround = flight.between(
                    first_attempt.start, next_attempt.stop
                )
                assert goaround is not None

                goaround = goaround.assign(
                    ILS=None,
                    airport=self.airport
                    if isinstance(self.airport, str)
                    else self.airport.icao,
                )
                goaround.data.loc[
                    goaround.data.timestamp <= first_attempt.stop, "ILS"
                ] = first_attempt.max("ILS")
                goaround.data.loc[
                    goaround.data.timestamp >= next_attempt.start, "ILS"
                ] = next_attempt.max("ILS")
                yield goaround

            first_attempt = next_attempt
