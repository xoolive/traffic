from typing import Iterator, Protocol

from typing_extensions import runtime_checkable

from ...core import Flight


@runtime_checkable
class ApplyBase(Protocol):
    """Classes following this protocol should implement an ``apply`` method
    which returns a :class:`~traffic.core.Flight`.
    """

    def apply(self, flight: Flight) -> Flight: ...


@runtime_checkable
class ApplyIteratorBase(Protocol):
    """Classes following this protocol should implement an ``apply`` method
    which returns an iterator of :class:`~traffic.core.Flight`.
    """

    def apply(self, flight: Flight) -> Iterator[Flight]: ...


@runtime_checkable
class ApplyOptionalBase(Protocol):
    """Classes following this protocol should implement an ``apply`` method
    which returns ``None`` or a :class:`~traffic.core.Flight`.
    """

    def apply(self, flight: Flight) -> None | Flight: ...
