from datetime import timedelta
from typing import Optional, cast

import pytest

from traffic.core import Flight, Traffic
from traffic.data import opensky
from traffic.data.samples import belevingsvlucht, lfbo_tma


@pytest.mark.timeout(300)
def test_flightlist() -> None:

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


@pytest.mark.timeout(300)
def test_history() -> None:

    t_aib: Optional[Traffic] = cast(
        Optional[Traffic],
        opensky.history(
            "2019-11-01 09:00",
            "2019-11-01 12:00",
            departure_airport="LFBO",
            arrival_airport="LFBO",
            callsign="AIB%",
            compress=True,
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
            compress=False,
        ),
    )
    assert t_tma is not None
    assert len(t_tma) == 33

    df = opensky.extended(
        "2019-11-11 10:00", "2019-11-11 10:10", bounds=lfbo_tma
    )

    t_decoded = t_tma.filter().query_ehs(df).eval(desc="", max_workers=4)
    assert len(t_decoded) == len(t_tma)


@pytest.mark.timeout(300)
def test_complex_queries() -> None:
    error_msg = "airport may not be set if arrival_airport is set"
    with pytest.raises(RuntimeError, match=error_msg):
        _ = opensky.history(
            start="2021-08-24 00:00",
            stop="2021-08-24 01:00",
            airport="ESSA",
            arrival_airport="EGLL",
        )
    # test that `limit` generate correct query
    t2 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        airport="ESSA",
        limit=3,
    )
    assert t2 is not None
    assert len(t2.data) == 3

    t2_1 = opensky.history(
        start="2021-08-24 09:00",
        stop="2021-08-24 09:10",
        airport="ESSA",
    )
    assert t2_1 is not None
    assert len(t2_1) == 23

    t3 = opensky.history(
        start="2021-08-24 09:00",
        stop="2021-08-24 09:10",
        arrival_airport="ESSA",
    )
    assert t3 is not None
    assert len(t3) == 13

    t4 = opensky.history(
        start="2021-08-24 11:32",
        stop="2021-08-24 11:42",
        departure_airport="ESSA",
        arrival_airport="EGLL",
    )
    assert t4 is not None
    assert len(t4) == 1
    flight = cast(Traffic, t4)["BAW777C"]
    assert flight is not None
    assert flight.icao24 == "400936"

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
        stop="2021-08-24 00:10",
        arrival_airport="ESSA",
        serials=-1408232560,
    )
    assert t6 is not None
    assert len(t6) == 1
    flight = cast(Traffic, t6)[0]
    assert flight is not None
    assert flight.callsign == "SAS6906"
    assert flight.icao24 == "4ca863"

    t7 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 00:10",
        serials=(-1408232560, -1408232534),
    )
    assert t7 is not None
    assert len(t7) == 12

    t8 = opensky.history(
        start="2021-08-24 09:00",
        stop="2021-08-24 09:10",
        departure_airport="ESSA",
        serials=(-1408232560, -1408232534),
    )
    assert t8 is not None
    assert len(t8) == 1
    flight = cast(Traffic, t8)[0]
    assert flight is not None
    assert flight.callsign == "LOT454"
    assert flight.icao24 == "489789"

    t9 = opensky.history(
        start="2021-08-24 09:00",
        stop="2021-08-24 09:10",
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
        serials=(-1408232560, -1408232534),
    )
    assert t9 is not None
    assert len(t9) == 9
    flight = cast(Traffic, t9)["SAS1136"]
    assert flight is not None
    assert flight.icao24 == "51110b"

    tA = opensky.history(
        start="2021-08-24 09:30",
        stop="2021-08-24 09:40",
        departure_airport="ESSA",
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
        serials=(-1408232560, -1408232534),
    )
    assert tA is not None
    assert len(tA) == 1
    flight = cast(Traffic, tA)[0]
    assert flight is not None
    assert flight.callsign == "THY5HT"
    assert flight.icao24 == "4bb1c5"

    tB = opensky.history(
        start="2021-08-24 09:45",
        stop="2021-08-24 09:50",
        departure_airport="ESSA",
        count=True,
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
        serials=(-1408232560, -1408232534),
    )
    assert tB is not None
    assert len(tB) == 1
    flight = cast(Traffic, tB)[0]
    assert flight is not None
    assert flight.callsign == "SAS69E"
    assert flight.icao24 == "4ac9e5"

    tC = opensky.history(
        start="2021-08-24 09:45",
        stop="2021-08-24 09:50",
        departure_airport="ESSA",
        count=True,
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
    )
    assert tC is not None
    assert len(tC) == 1
    flight = cast(Traffic, tC)[0]
    assert flight is not None
    assert flight.callsign == "SAS69E"
    assert flight.icao24 == "4ac9e5"

    tD = opensky.history(
        start="2021-08-24 09:45",
        stop="2021-08-24 09:50",
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
    )
    assert tD is not None
    assert len(tD) == 9


# @pytest.mark.timeout(300)
def test_timebuffer() -> None:

    f = cast(
        Flight,
        opensky.history(
            "2021-09-06 14:00",
            "2021-09-07 16:00",
            callsign="SAS44A",
            airport="ESSA",
            time_buffer="1H",
            return_flight=True,
        ),
    )
    assert f is not None

    g = f.resample("1s").cumulative_distance().query("compute_gs > 5")
    assert g is not None
    h = g.on_taxiway("ESSA").max(key="stop")
    assert h is not None
    assert h.taxiway_max == "Z"


@pytest.mark.timeout(300)
def test_rawdata() -> None:

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
    assert f.max("altitude") == 11050
