from typing import Protocol, runtime_checkable

from ...core.flight import Flight


@runtime_checkable
class PredictBase(Protocol):
    def predict(self, flight: Flight) -> Flight: ...
