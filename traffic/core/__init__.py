# flake8: noqa
"""
It is crucial that the imports do not change order,
hence the following line:
isort:skip_file 
"""

import logging
import sys
from typing import Any, Dict, Optional

# WARNING!! Don't change order of import in this file
from .flight import Flight
from .iterator import FlightIterator
from .traffic import Traffic
from .airspace import Airspace
from .sv import StateVectors
from .flightplan import FlightPlan


def loglevel(mode: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, mode))


def faulty_flight(exc=None) -> Dict[str, Any]:
    if exc is None:
        exc = sys.last_traceback

    tb = exc.tb_next.tb_next
    while tb is not None:
        loc = tb.tb_frame.f_locals
        if any(isinstance(x, Flight) for x in loc.values()):
            return loc
        tb = tb.tb_next

    return dict()
