from typing import Protocol

from ...core.flight import Flight


class PredictBase(Protocol):
    def predict(self, flight: Flight) -> Flight: ...
