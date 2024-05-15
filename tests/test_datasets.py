from traffic.data.datasets.scat import SCAT

import pandas as pd


def test_scat() -> None:
    s = SCAT("scat20161015_20161021.zip", nflights=10)
    assert len(s.traffic) == 10
    assert s.flight_plans.flight_id.nunique() == 10
    assert s.clearances.flight_id.nunique() == 10


def test_scat_waypoints() -> None:
    s = SCAT("scat20161015_20161021.zip", nflights=10, include_waypoints=True)
    assert isinstance(s.waypoints, pd.DataFrame)


def test_scat_weather() -> None:
    s = SCAT("scat20161015_20161021.zip", nflights=10, include_weather=True)
    assert isinstance(s.weather, pd.DataFrame)
