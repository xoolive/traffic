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
    assert len(s.waypoints) == 15871
    assert set(s.waypoints.columns) == {
        "latitude",
        "longitude",
        "name",
        "center",
    }
    aa212 = s.waypoints[s.waypoints["name"] == "AA212"]
    assert len(aa212) == 1
    assert aa212["latitude"].item() == 58.4902778
    assert aa212["longitude"].item() == 14.4866667
    assert aa212["center"].item() == "ESMM"

    # KERAX is present for both centers
    kerax = s.waypoints[s.waypoints["name"] == "KERAX"]
    assert set(kerax["center"].values) == {"ESMM", "ESOS"}
    assert kerax.iloc[0]["latitude"] == kerax.iloc[1]["latitude"] == 50.475
    assert kerax.iloc[0]["longitude"] == kerax.iloc[1]["longitude"] == 9.5819444


def test_scat_weather() -> None:
    s = SCAT("scat20161015_20161021.zip", nflights=10, include_weather=True)
    assert isinstance(s.weather, pd.DataFrame)
    assert not s.weather.isna().any().max()
    assert len(s.weather) == 1519310
    assert set(s.weather.columns) == {
        "altitude",
        "latitude",
        "longitude",
        "temperature",
        "timestamp",
        "wind_direction",
        "wind_speed",
    }
    assert isinstance(s.weather["timestamp"].dtype, pd.DatetimeTZDtype)

    # compare measurement for a specific timestamp
    ts = pd.to_datetime("2016-10-14 10:30:00+00:00")  # noqa: F841
    measurement = s.weather.query(
        "timestamp == @ts & altitude == 50 & latitude == 42.5 & longitude == 60"
    )
    assert len(measurement) == 1
    assert measurement["temperature"].item() == 4
    assert measurement["wind_direction"].item() == 166
    assert measurement["wind_speed"].item() == 16
