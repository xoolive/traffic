# ruff: noqa: I001
"""
It is crucial that the imports do not change order,
hence the following line:
# ruff: noqa: I001
"""

import logging
import sys
from types import TracebackType
from typing import Any, Dict, Optional

# WARNING!! Don't change order of import in this file
from .flight import Flight
from .iterator import FlightIterator
from .traffic import Traffic
from .lazy import LazyTraffic
from .airspace import Airspace
from .sv import StateVectors
from .flightplan import FlightPlan
from .leaflet import monkey_patch

__all__ = [
    "Flight",
    "FlightIterator",
    "Traffic",
    "Airspace",
    "StateVectors",
    "FlightPlan",
    "LazyTraffic",
    "loglevel",
    "faulty_flight",
]


def loglevel(mode: str) -> None:
    """
    Changes the log level of the libraries root logger.

    :param mode:
        New log level.
    """
    _log = logging.getLogger("traffic")
    if not any(isinstance(h, logging.StreamHandler) for h in _log.handlers):
        _log.addHandler(logging.StreamHandler())
        _log.info("Setting a default StreamHandler")
    _log.setLevel(getattr(logging, mode))


def faulty_flight(exc: Optional[TracebackType] = None) -> Dict[str, Any]:
    if exc is None:
        exc = sys.last_traceback

    if exc is None:
        raise RuntimeError("No exception encountered")

    assert exc.tb_next is not None
    tb = exc.tb_next.tb_next
    while tb is not None:
        loc = tb.tb_frame.f_locals
        if any(isinstance(x, Flight) for x in loc.values()):
            return loc
        tb = tb.tb_next

    return dict()


monkey_patch()  # leaflet
