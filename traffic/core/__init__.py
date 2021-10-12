# flake8: noqa
"""
It is crucial that the imports do not change order,
hence the following line:
isort:skip_file 
"""

import logging
import sys
from types import TracebackType
from typing import Any, Dict, Optional

# WARNING!! Don't change order of import in this file
from .flight import Flight
from .iterator import FlightIterator
from .traffic import Traffic
from .airspace import Airspace
from .sv import StateVectors
from .flightplan import FlightPlan

__all__ = [
    "Flight",
    "FlightIterator",
    "Traffic",
    "Airspace",
    "StateVectors",
    "FlightPlan",
    "loglevel",
    "faulty_flight",
]


def loglevel(mode: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, mode))


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
