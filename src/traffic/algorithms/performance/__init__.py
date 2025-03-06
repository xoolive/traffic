from typing import Protocol, runtime_checkable

from ...core import Flight


@runtime_checkable
class EstimatorBase(Protocol):
    def estimate(self, flight: Flight) -> Flight: ...
