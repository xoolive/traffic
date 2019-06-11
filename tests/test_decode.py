import sys
from pathlib import Path

import pytest

from traffic.core import Flight, Traffic
from traffic.data import opensky
from traffic.data.samples import collections


def long_enough(flight: Flight) -> bool:
    return len(flight) > 100


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_decode():

    opensky.cache_dir = Path(__file__).parent.parent / "data" / "opensky_cache"

    # with zipfile.ZipFile(opensky.cache_dir / "opensky_cache.zip") as zfile:
    #     zfile.extractall(path=opensky.cache_dir)

    switzerland: Traffic = getattr(collections, "switzerland")

    tap_switzerland = (
        switzerland.query('callsign.str.startswith("TAP127")')
        .filter_if(long_enough)
        .query_opensky()
        .resample("1s")
        .query_ehs()
        .filter(selected_mcp=23)
        .filter(altitude=53, selected_mcp=53, roll=53, heading=53)
        .resample("1s")
        .eval(desc="")
    )

    # BDS 4,0
    for f in tap_switzerland:
        # This is only safe en route. Even if the selected MCP altitude
        # changes, altitude should be between min/max (modulo data errors)
        assert all(
            (f.min("selected_mcp") - 100 <= f.data.altitude)
            & (f.data.altitude <= f.max("selected_mcp") + 100)
        )

    # BDS 5,0 and BDS 6,0
    for f in tap_switzerland:
        # An aircraft should turn to the side it is rolling
        f = f.assign(diff_heading=lambda df: df.heading.diff() * df.roll)
        assert sum(f.data.diff_heading + 1 < 0) / len(f) < 1e-3
