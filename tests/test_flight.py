import sys

import pytest
from traffic.core import Flight
from traffic.data.samples import featured


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_landing_airport() -> None:
    flight: Flight = getattr(featured, "belevingsvlucht")
    assert flight.guess_landing_airport().airport.icao == "EHAM"


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_landing_runway() -> None:
    flight: Flight = getattr(featured, "belevingsvlucht")
    assert flight.guess_landing_runway().name == "06"
