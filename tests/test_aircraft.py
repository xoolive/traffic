import pytest

import pandas as pd
from traffic.data import aircraft
from traffic.data.basic.aircraft import Aircraft

aircraft.download_junzis()  # make tests based on Junzi's (light) database


def test_getter() -> None:
    a = aircraft["PH-BHA"]
    assert a is not None
    a_dict = aircraft.get("PH-BHA")
    assert a_dict is not None
    assert a_dict["icao24"] == "4851ad"
    assert a_dict["typecode"] == "B789"

    a = aircraft["39b415"]
    assert a is not None
    a_dict = aircraft.get("39b415")
    assert a_dict is not None
    assert a_dict["registration"] == "F-HNAV"
    assert a_dict["typecode"] == "BE20"


@pytest.mark.skipif(aircraft.data is not None, reason="OpenSky database")
def test_opensky_dl() -> None:
    aircraft.download_opensky()
    assert aircraft.data is not None


def test_operator() -> None:
    df = aircraft.operator("British Antarctic Survey")
    assert df is not None
    assert df.data.registration.str.startswith("VP-F").all()


def test_stats() -> None:
    df = aircraft.stats("Lufthansa")
    assert df is not None
    assert df.shape[0] > 0
    assert df.icao24.sum() > 200


def test_query() -> None:
    df = aircraft.query(operator="KLM", model="b789")
    assert df is not None
    assert (df.data.typecode == "B789").all()
    assert df.data.registration.str.startswith("PH-").all()


def test_country() -> None:
    reg = "9T-ABC"
    act = {
        "icao24": ["000001"],
        "registration": [reg],
    }
    a = Aircraft(pd.DataFrame(act)).get(reg)
    assert a is not None
    assert a["country"] == "Democratic Republic of the Congo"
