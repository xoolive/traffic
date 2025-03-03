from typing import Iterator, Protocol

from ...core import Flight


class ApplyBase(Protocol):
    """Classes following this protocol should implement an `apply` method
    which returns a Flight.
    """

    def apply(self, flight: Flight) -> Flight: ...


class ApplyIteratorBase(Protocol):
    """Classes following this protocol should implement an `apply` method
    which returns an iterator of Flight.
    """

    def apply(self, flight: Flight) -> Iterator[Flight]: ...


class ApplyOptionalBase(Protocol):
    """Classes following this protocol should implement an `apply` method
    which returns None or a Flight.
    """

    def apply(self, flight: Flight) -> None | Flight: ...
