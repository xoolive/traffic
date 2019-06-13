import sys
from pathlib import Path

from traffic.core import Flight, Traffic
from traffic.data.samples import collections, get_flight


def get_sample(module, name: str):
    if sys.version_info >= (3, 7):
        return getattr(module, name)
    path = Path(module.__file__).parent
    return get_flight(name, path)


def test_properties() -> None:
    switzerland: Traffic = get_sample(collections, "switzerland")
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


def high_altitude(flight: Flight) -> bool:
    return flight.min("altitude") > 35000


def test_chaining() -> None:
    switzerland: Traffic = get_sample(collections, "switzerland")
    sw_filtered = (
        switzerland.assign_id()
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
    assert sw_filtered.data.shape[0] == 86399
    assert min(len(f) for f in sw_filtered) == 61
    assert sw_filtered.data.altitude.max() == 47000.0
