import zipfile

import pytest

from traffic.data import airports, runways

skip_runways = False

try:
    _ = runways.runways
except zipfile.BadZipFile:
    skip_runways = True


def test_getter() -> None:
    airport = airports["LHR"]
    assert airport is not None
    assert airport.icao == "EGLL"
    assert airport.iata == "LHR"
    assert airport.country == "United Kingdom"
    assert airport.name == "London Heathrow Airport"
    lat, lon = airport.latlon
    assert max(abs(lat - 51.471626), abs(lon + 0.467081)) < 1e-2


def test_search() -> None:
    denmark = airports.search("denmark")
    assert denmark is not None
    denmark_large = denmark.query('type == "large_airport"')
    assert denmark_large is not None
    assert denmark_large.data.icao.str.startswith("EK").all()

    italy = airports.search("ITALY")
    assert italy is not None
    italy_large = italy.query('type == "large_airport"')
    assert italy_large is not None
    assert italy_large.data.icao.str.startswith("LI").all()


@pytest.mark.skipif(skip_runways, reason="Failed to download runway data")
def test_runway_list() -> None:
    airport = airports["TLS"]
    assert airport is not None
    assert airport.runways is not None
    rwy_list = set(t.name for t in airport.runways.list)
    assert rwy_list == {"14L", "14R", "32L", "32R"}


@pytest.mark.skipif(skip_runways, reason="Failed to download runway data")
def test_runway_bearing() -> None:
    for apt_name in ["EHAM", "EDDF", "LFPG", "KLAX", "KSFO", "RJTT"]:
        airport = airports[apt_name]
        if airport is None or airport.runways is None:
            # Robustness against airports being maliciously edited out
            # of OurAirports database
            continue
        for runway in airport.runways.list:
            delta = abs(int(runway.name[:2]) * 10 - runway.bearing)
            if delta > 180:
                delta = 360 - delta
            # It can be as big as 25 degrees with parallel runways!
            assert delta < 25, f"Error with airport {apt_name} {runway.name}"
