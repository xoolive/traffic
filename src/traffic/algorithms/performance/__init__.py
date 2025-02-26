from typing import Protocol

from ...core import Flight


class EstimatorBase(Protocol):
    def estimate(self, flight: Flight) -> Flight: ...
