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
        )
    # test that `limit` generate correct query
    t2 = opensky.history(
        start="2021-08-24 00:00",
        stop="2021-08-24 01:00",
        airport="ESSA",
        limit=3,
    )
    assert len(t2.data) == 3

    t2_1 = opensky.history(
        start="2021-08-24 09:00",
        stop= "2021-08-24 09:10",
        airport="ESSA",
    )
    assert len(t2_1) == 23

    t3 = opensky.history(
        start="2021-08-24 09:00",
        stop= "2021-08-24 09:10",
        arrival_airport="ESSA",
    )
    assert len(t3) == 13

    t4 = opensky.history(
        start="2021-08-24 11:32",
        stop= "2021-08-24 11:42",
        departure_airport="ESSA",
        arrival_airport="EGLL",
    )
    assert len(t4) == 1
    flight = t4["BAW777C"]
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
        stop= "2021-08-24 00:10",
        arrival_airport="ESSA",
        serials=-1408232560,
    )
    assert len(t6) == 1
    flight = t6[0]
    assert flight is not None
    assert flight.callsign == "SAS6906"    
    assert flight.icao24 == "4ca863"    

    t7 = opensky.history(
        start="2021-08-24 00:00",
        stop= "2021-08-24 00:10",
        serials=(-1408232560, -1408232534),
    )
    assert len(t7) == 12

    t8 = opensky.history(
        start="2021-08-24 09:00",
        stop= "2021-08-24 09:10",
        departure_airport="ESSA",
        serials=(-1408232560, -1408232534),
    )
    assert len(t8) == 1
    flight = t8[0]
    assert flight is not None
    assert flight.callsign == "LOT454"    
    assert flight.icao24 == "489789"    

    t9 = opensky.history(
        start="2021-08-24 09:00",
        stop= "2021-08-24 09:10",
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
        serials=(-1408232560, -1408232534),
    )
    assert len(t9) == 9
    flight = t9["SAS1136"]
    assert flight is not None
    assert flight.icao24 == "51110b"    

    tA = opensky.history(
        start="2021-08-24 09:30",
        stop= "2021-08-24 09:40",
        departure_airport="ESSA",
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
        serials=(-1408232560, -1408232534),
    )
    assert len(tA) == 1
    flight = tA[0]
    assert flight is not None
    assert flight.callsign == "THY5HT"    
    assert flight.icao24 == "4bb1c5"    

    tB = opensky.history(
        start="2021-08-24 09:45",
        stop= "2021-08-24 09:50",
        departure_airport="ESSA",
        count=True,
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
        serials=(-1408232560, -1408232534),
    )
    assert len(tB) == 1
    flight = tB[0]
    assert flight is not None
    assert flight.callsign == "SAS69E"    
    assert flight.icao24 == "4ac9e5"    

    tC = opensky.history(
        start="2021-08-24 09:45",
        stop= "2021-08-24 09:50",
        departure_airport="ESSA",
        count=True,
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
    )
    assert len(tC) == 1
    flight = tC[0]
    assert flight is not None
    assert flight.callsign == "SAS69E"    
    assert flight.icao24 == "4ac9e5"    

    tD = opensky.history(
        start="2021-08-24 09:45",
        stop= "2021-08-24 09:50",
        bounds=[17.8936, 59.6118, 17.9894, 59.6716],
    )
    assert len(tD) == 9

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
