from datetime import timedelta
from typing import Optional, cast

import pytest

from traffic.core import Traffic
from traffic.data import opensky
from traffic.data.samples import belevingsvlucht, lfbo_tma


def test_flightlist():

    l_aib = opensky.flightlist(
        "2019-11-01",
        departure_airport="LFBO",
        arrival_airport="LFBO",
        callsign="AIB%",
    )
    assert l_aib.shape[0] == 2
    assert all(l_aib.callsign.str.startswith("AIB"))

    l_44017c = opensky.flightlist("2019-11-01", icao24="44017c")
    assert all(l_44017c.callsign.str.startswith("EJU"))


def test_history():

    t_aib: Optional[Traffic] = cast(
        Optional[Traffic],
        opensky.history(
            "2019-11-01 09:00",
            "2019-11-01 12:00",
            departure_airport="LFBO",
            arrival_airport="LFBO",
            callsign="AIB%",
        ),
    )
    assert t_aib is not None

    flight = t_aib["AIB04FI"]
    assert flight is not None
    assert flight.icao24 == "388dfb"

    t_tma: Optional[Traffic] = cast(
        Optional[Traffic],
        opensky.history(
            "2019-11-11 10:00",
            "2019-11-11 10:10",
            bounds=lfbo_tma,
            serials=1433801924,
        ),
    )
    assert t_tma is not None
    assert len(t_tma) == 33

    df = opensky.extended(
        "2019-11-11 10:00", "2019-11-11 10:10", bounds=lfbo_tma
    )

    t_decoded = t_tma.filter().query_ehs(df).eval(desc="", max_workers=4)
    assert len(t_decoded) == len(t_tma)


def test_complex_queries():
    error_msg = "airport may not be set if arrival_airport is set"
    with pytest.raises(RuntimeError, match=error_msg):
        _ = opensky.history(
            start="2021-08-24 00:00",
            stop="2021-08-24 01:00",
            airport="ESSA",
            arrival_airport="EGLL",
            limit=3,
        )

    t2 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        airport="ESSA",
        limit=3,
    )
    assert t2 is not None

    t3 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        arrival_airport="ESSA",
        limit=3,
    )
    assert t3 is not None

    t4 = opensky.history(
        start="2021-08-24 00:00",
        departure_airport="ESSA",
        arrival_airport="EGLL",
        limit=3,
    )
    assert t4 is not None

    with pytest.raises(RuntimeError, match=error_msg):
        _ = opensky.history(
            start="2021-08-24 00:00",
            stop="2021-08-24 01:00",
            airport="ESSA",
            arrival_airport="EGLL",
            limit=3,
        )

    t6 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        arrival_airport="ESSA",
        serials=-1408232560,
        limit=3,
    )
    assert t6 is not None

    t7 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        serials=(-1408232560, -1408232534),
        limit=3,
    )
    assert t7 is not None

    t8 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        departure_airport="ESSA",
        serials=(-1408232560, -1408232534),
        limit=3,
    )
    assert t8 is not None


def test_rawdata():

    r = opensky.rawdata(
        belevingsvlucht.start,
        belevingsvlucht.start + timedelta(minutes=10),
        icao24=belevingsvlucht.icao24,
    )

    assert r is not None

    t = r.decode("EHAM", uncertainty=True)
    assert t is not None

    f = t["484506"]
    assert f is not None

    assert len(f) == 2751
    assert f.max("altitude") == 11050
