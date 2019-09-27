import pandas as pd

from traffic.core import Flight
from traffic.data import eurofirs
from traffic.data.samples import switzerland


def test_properties() -> None:
    assert len(switzerland) == 1244
    assert f"{switzerland.start_time}" == "2018-08-01 05:00:00+00:00"
    assert f"{switzerland.end_time}" == "2018-08-01 21:59:50+00:00"

    # TODO change @lru_cache on @property, rename Traffic.aircraft
    assert len(switzerland.callsigns) == 1243  # type: ignore
    assert len(switzerland.aircraft) == 842  # type: ignore

    handle = switzerland["DLH02A"]
    assert handle is not None
    assert handle.aircraft == "3c6645 / D-AIRE (A321)"

    handle = switzerland["4baa61"]
    assert handle is not None
    assert handle.callsign == "THY7WR"

    selected = max(switzerland, key=lambda flight: flight.min("altitude"))
    assert selected.flight_id is None
    assert selected.min("altitude") == 47000.0
    assert selected.icao24 == "aab6c0"


def test_index() -> None:
    df = pd.DataFrame.from_records(
        [
            {
                "icao24": "500142",
                "callsign": "T7STK",
                "start": "2018-08-01 15:00",
                "stop": "2018-08-01 16:00",
            },
            {
                "icao24": "4068cb",
                "callsign": "EXS33W",
                "start": None,
                "stop": None,
            },
            {
                "icao24": "4009f9",
                "callsign": "BAW585E",
                "start": None,
                "stop": "2018-08-01 17:00",
            },
        ]
    )

    assert len(switzerland[df]) == 3
    assert switzerland[df.iloc[0]] is not None
    assert len(switzerland[["EXS33W", "4009f9"]]) == 4


def test_aircraft() -> None:
    assert set(
        f.max("typecode")
        for f in switzerland[["EXS33W", "4009f9"]].aircraft_data()
    ) == {"A320", "B733"}


def high_altitude(flight: Flight) -> bool:
    return flight.min("altitude") > 35000


def test_chaining() -> None:
    sw_filtered = (
        switzerland.between("2018-08-01", "2018-08-02")  # type: ignore
        .inside_bbox(eurofirs["LSAS"])
        .assign_id()
        .filter_if(high_altitude)
        .resample("10s")
        .filter()
        .filter(altitude=53)
        .unwrap()
        .airborne()
        .eval(max_workers=4)
    )
    flight_id: str = sw_filtered.flight_ids.pop()
    handle = sw_filtered[flight_id]
    assert handle is not None
    assert handle.callsign == flight_id.split("_")[0]
    assert len(sw_filtered) == 784
    assert sw_filtered.data.shape[0] > 80000
    assert min(len(f) for f in sw_filtered) == 60
    assert sw_filtered.data.altitude.max() == 47000.0
