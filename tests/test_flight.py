import sys
import zipfile

import pytest

from traffic.core import Flight, Traffic
from traffic.data import eurofirs, runways, samples
from traffic.data.samples import featured

# This part only serves on travis when the downloaded file is corrupted
# This shouldn't happen much as caching is now activated.
skip_runways = False

try:
    _ = runways.runways
except zipfile.BadZipFile:
    skip_runways = True


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_properties() -> None:
    flight: Flight = getattr(samples, "belevingsvlucht")
    assert len(flight) == 16005
    assert flight.min("altitude") == -59  # Welcome to the Netherlands!
    assert flight.max("altitude") == 18025
    assert f"{flight.start}" == "2018-05-30 15:21:38+00:00"
    assert f"{flight.stop}" == "2018-05-30 20:22:56+00:00"
    assert flight.callsign == "TRA051"
    assert flight.title == "TRA051"
    assert flight.icao24 == "484506"
    assert flight.registration == "PH-HZO"
    assert flight.typecode == "B738"
    assert flight.aircraft == "484506 / PH-HZO (B738)"
    assert flight.flight_id is None

    traffic: Traffic = getattr(featured, "traffic")
    flight = traffic["belevingsvlucht"]  # type: ignore
    assert flight is not None
    assert flight.flight_id == "belevingsvlucht"


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_emptydata() -> None:
    airbus_tree: Flight = getattr(samples, "airbus_tree")
    assert airbus_tree.registration == "F-WWAE"
    assert airbus_tree.typecode == "A388"


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_iterators() -> None:
    flight: Flight = getattr(samples, "belevingsvlucht")
    assert min(flight.timestamp) == flight.start
    assert max(flight.timestamp) == flight.stop
    assert min(flight.coords)[0] == flight.min("longitude")
    assert max(flight.coords)[0] == flight.max("longitude")

    max_time = max(flight.coords4d())
    last_point = flight.at()
    assert last_point is not None
    assert max_time[1] == last_point.longitude
    assert max_time[2] == last_point.latitude
    assert max_time[3] == last_point.altitude

    max_xy_time = list(flight.xy_time)[-1]
    assert max_xy_time[0] == last_point.longitude
    assert max_xy_time[1] == last_point.latitude
    assert max_xy_time[2] == last_point.timestamp.to_pydatetime().timestamp()


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_time_methods() -> None:
    flight: Flight = getattr(samples, "belevingsvlucht")
    assert f"{flight.first(minutes=10).stop}" == "2018-05-30 15:31:37+00:00"
    assert f"{flight.last(minutes=10).start}" == "2018-05-30 20:12:57+00:00"

    # between is a combination of before and after
    before_after = flight.before("2018-05-30 19:00").after("2018-05-30 18:00")
    between = flight.between("2018-05-30 18:00", "2018-05-30 19:00")

    # flight comparison made by distance computation
    assert before_after.distance(between).lateral.sum() < 1e-6
    assert between.distance(before_after).vertical.sum() < 1e-6

    # test of at() method and equality on the positions
    t = "2018-05-30 18:30"
    assert (between.at(t) == before_after.at(t)).all()  # type: ignore


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_geometry() -> None:
    flight: Flight = getattr(samples, "belevingsvlucht")
    xy_length = flight.project_shape().length / 1852  # in nm
    last_pos = flight.cumulative_distance().at()
    assert last_pos is not None
    cumdist = last_pos.cumdist
    assert abs(xy_length - cumdist) / xy_length < 1e-3

    simplified = flight.simplify(1e3)
    assert len(simplified) < len(flight)
    xy_length_s = simplified.project_shape().length / 1852
    assert xy_length_s < xy_length

    assert flight.intersects(eurofirs["EHAA"])
    assert flight.intersects(eurofirs["EHAA"].flatten())
    assert not flight.intersects(eurofirs["LFBB"])

    airbus_tree: Flight = getattr(samples, "airbus_tree")
    clip_dk = airbus_tree.clip(eurofirs["EKDK"])
    assert clip_dk is not None
    assert clip_dk.duration < flight.duration

    clip_gg = airbus_tree.clip(eurofirs["EDGG"])
    assert clip_gg is not None
    assert clip_gg.duration < flight.duration

    clip_mm = airbus_tree.clip(eurofirs["EDMM"])
    assert clip_mm is not None
    assert clip_mm.duration < flight.duration


def test_filtering_methods() -> None:
    # TODO
    # filter(), resample(freq, nb_points), comet()
    pass


def test_opensky() -> None:
    # TODO
    # query_opensky(), query_ehs(),
    pass


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py37")
def test_landing_airport() -> None:
    # TODO refactor/rethink the returned type
    flight: Flight = getattr(samples, "belevingsvlucht")
    assert flight.guess_landing_airport().airport.icao == "EHAM"

    airbus_tree: Flight = getattr(samples, "airbus_tree")
    assert airbus_tree.guess_landing_airport().airport.icao == "EDHI"


@pytest.mark.skipif(sys.version_info < (3, 7) or skip_runways, reason="py37")
def test_landing_runway() -> None:
    # TODO refactor/rethink the returned type
    flight: Flight = getattr(samples, "belevingsvlucht")
    assert flight.guess_landing_runway().name == "06"

    airbus_tree: Flight = getattr(samples, "airbus_tree")
    assert airbus_tree.guess_landing_runway().name == "23"
