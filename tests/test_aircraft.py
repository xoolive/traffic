import pytest

from traffic.data import aircraft


def test_getter() -> None:
    a = aircraft["PH-BHA"]
    assert a.icao24.iloc[0] == "4851ad"
    assert a.typecode.iloc[0] == "B789"
    a = aircraft["39b415"]
    assert a.registration.iloc[0] == "F-HNAV"
    assert a.typecode.iloc[0] == "BE20"


@pytest.mark.skipif(aircraft.opensky_db is not None, reason="OpenSky database")
def test_opensky_dl() -> None:
    aircraft.download_opensky()
    assert aircraft.opensky_db is not None


@pytest.mark.skipif(aircraft.opensky_db is None, reason="OpenSky database")
def test_operator() -> None:
    df = aircraft.operator("penguin")
    assert df.operator.iloc[0] == "British Antarctic Survey"
    assert df.operatoricao.iloc[0] == "BAN"
    assert df.registration.str.startswith("VP-F").all()


@pytest.mark.skipif(aircraft.opensky_db is None, reason="OpenSky database")
def test_stats() -> None:
    df = aircraft.stats("speedbird")
    # well let's say they must be enough...
    assert df.shape[0] > 10
    assert df.icao24.sum() > 200


def test_query() -> None:
    df = aircraft.query(operator="KLM", model="b789")
    assert (df.typecode == "B789").all()
    assert df.registration.str.startswith("PH-").all()

    df = aircraft.query(registration="^F-ZB", model="S2P")
    assert (df.typecode == "S2P").all()
    assert df.registration.str.startswith("F-ZB").all()
