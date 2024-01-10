from traffic.data.datasets.scat import SCAT


def test_scat() -> None:
    s = SCAT("scat20161015_20161021.zip", nflights=10)
    assert len(s.traffic) == 10
    assert s.flight_plans.flight_id.nunique() == 10
    assert s.clearances.flight_id.nunique() == 10
