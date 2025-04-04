from typing import cast

import pytest
from cartes.crs import CH1903p  # type: ignore

from traffic.core import Traffic
from traffic.data.samples import collections, get_sample


@pytest.mark.slow
def test_cpa() -> None:
    switzerland = cast(Traffic, get_sample(collections, "switzerland"))

    between = switzerland.between("2018-08-01 12:00", "2018-08-01 14:00")
    assert between is not None
    smaller = between.assign_id().eval()
    assert smaller is not None

    cpa = smaller.closest_point_of_approach(
        lateral_separation=10 * 1852,
        vertical_separation=2000,
        projection=CH1903p(),
        round_t="10 min",
    )

    assert cpa is not None
    separation = dict(lateral_separation=5, vertical_separation=1000)
    res15 = cpa.aggregate(**separation)["0a0075"].query("aggregated < 1.5")

    assert res15 is not None
    res = res15.min("aggregated")

    callsigns = {*res.data.callsign_x, *res.data.callsign_y}
    assert callsigns == {"BAW2591", "BAW605", "DAH2062", "EZY54UC"}
    assert len(res.flight_ids()) == 4
