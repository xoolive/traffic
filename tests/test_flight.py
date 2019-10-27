# fmt: off

import sys
import zipfile

import pandas as pd
import pytest

from traffic.algorithms.douglas_peucker import douglas_peucker
from traffic.core import Flight, Traffic
from traffic.data import eurofirs, navaids, runways
from traffic.data.samples import (airbus_tree, belevingsvlucht, calibration,
                                  featured, get_sample)

# fmt: on

# This part only serves on travis when the downloaded file is corrupted
# This shouldn't happen much as caching is now activated.
skip_runways = False

try:
    _ = runways.runways
except zipfile.BadZipFile:
    skip_runways = True


def test_properties() -> None:
    flight = belevingsvlucht
    assert len(flight) == 16005
    assert flight.min("altitude") == -59  # Welcome to the Netherlands!
    assert flight.max("altitude") == 18025
    assert flight.last(minutes=20).mean("vertical_rate") < -500
    assert f"{flight.start}" == "2018-05-30 15:21:38+00:00"
    assert f"{flight.stop}" == "2018-05-30 20:22:56+00:00"
    assert flight.callsign == "TRA051"
    assert flight.title == "TRA051"
    flight2 = flight.assign(number="FAKE", flight_id="belevingsvlucht")
    assert flight2.title == "TRA051 / FAKE (belevingsvlucht)"
    assert flight.icao24 == "484506"
    assert flight.registration == "PH-HZO"
    assert flight.typecode == "B738"
    assert flight.aircraft == "484506 / PH-HZO (B738)"
    assert flight.flight_id is None


@pytest.mark.skipif(sys.version_info < (3, 7), reason="py36")
def test_get_traffic() -> None:
    traffic: Traffic = get_sample(featured, "traffic")
    assert "belevingsvlucht" in traffic.flight_ids  # type: ignore


def test_emptydata() -> None:
    assert airbus_tree.registration == "F-WWAE"
    assert airbus_tree.typecode == "A388"


def test_iterators() -> None:
    flight = belevingsvlucht
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


def test_time_methods() -> None:
    flight = belevingsvlucht
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

    assert flight.longer_than("1 minute")
    assert flight.shorter_than("1 day")
    assert not flight.shorter_than(flight.duration)
    assert not flight.longer_than(flight.duration)
    assert flight.shorter_than(flight.duration, strict=False)
    assert flight.longer_than(flight.duration, strict=False)

    b = flight.between(None, "2018-05-30 19:00", strict=False)
    a = flight.between("2018-05-30 18:00", None, strict=False)

    assert a.shorter_than(flight.duration)
    assert b.shorter_than(flight.duration)

    shorter = flight.query("altitude < 100").max_split(10)
    assert shorter is not None
    assert shorter.duration < pd.Timedelta("6 minutes")

    point = flight.at_ratio(0.5)
    assert point is not None
    assert flight.start < point.timestamp < flight.stop

    point = flight.at_ratio(0)
    assert point is not None
    assert point.timestamp == flight.start

    point = flight.at_ratio(1)
    assert point is not None
    assert point.timestamp == flight.stop


def test_bearing() -> None:
    ajaccio: Flight = get_sample(calibration, "ajaccio")

    vor = navaids.extent(ajaccio)["AJO"]
    assert vor is not None
    gen = ajaccio.bearing(vor).query("bearing.diff().abs() < .01").split("1T")
    assert (
        sum(1 for chunk in gen if chunk.duration > pd.Timedelta("5 minutes"))
        == 7
    )


def test_geometry() -> None:
    flight: Flight = get_sample(featured, "belevingsvlucht")
    xy_length = flight.project_shape().length / 1852  # in nm
    last_pos = flight.cumulative_distance().at()
    assert last_pos is not None
    cumdist = last_pos.cumdist
    assert abs(xy_length - cumdist) / xy_length < 1e-3

    simplified = flight.simplify(1e3)
    assert len(simplified) < len(flight)
    xy_length_s = simplified.project_shape().length / 1852
    assert xy_length_s < xy_length

    simplified_3d = flight.simplify(1e3, altitude="altitude")
    assert len(simplified) < len(simplified_3d) < len(flight)

    assert flight.intersects(eurofirs["EHAA"])
    assert flight.intersects(eurofirs["EHAA"].flatten())
    assert not flight.intersects(eurofirs["LFBB"])

    airbus_tree: Flight = get_sample(featured, "airbus_tree")
    clip_dk = airbus_tree.clip(eurofirs["EKDK"])
    assert clip_dk is not None
    assert clip_dk.duration < flight.duration

    clip_gg = airbus_tree.clip(eurofirs["EDGG"])
    assert clip_gg is not None
    assert clip_gg.duration < flight.duration

    clip_mm = airbus_tree.clip(eurofirs["EDMM"])
    assert clip_mm is not None
    assert clip_mm.duration < flight.duration


def test_clip_point() -> None:
    records = [
        {
            "timestamp": pd.Timestamp("2019-07-02 15:02:30+0000", tz="UTC"),
            "longitude": -1.3508333333333333,
            "latitude": 46.5,
            "altitude": 36000,
            "callsign": "WZZ1066",
            "flight_id": "231619151",
            "icao24": "471f52",
        },
        {
            "timestamp": pd.Timestamp("2019-07-02 15:04:42+0000", tz="UTC"),
            "longitude": -1.00055555,
            "latitude": 46.664444450000005,
            "altitude": 36000,
            "callsign": "WZZ1066",
            "flight_id": "231619151",
            "icao24": "471f52",
        },
        {
            "timestamp": pd.Timestamp("2019-07-02 15:15:52+0000", tz="UTC"),
            "longitude": 0.5097222166666667,
            "latitude": 47.71388888333333,
            "altitude": 36000,
            "callsign": "WZZ1066",
            "flight_id": "231619151",
            "icao24": "471f52",
        },
    ]
    flight = Flight(pd.DataFrame.from_records(records))
    assert flight.clip(eurofirs["LFBB"]) is None


def test_landing_airport() -> None:
    # TODO refactor/rethink the returned type
    flight: Flight = get_sample(featured, "belevingsvlucht")
    assert flight.guess_landing_airport().airport.icao == "EHAM"

    airbus_tree: Flight = get_sample(featured, "airbus_tree")
    assert airbus_tree.guess_landing_airport().airport.icao == "EDHI"


@pytest.mark.skipif(skip_runways, reason="no runways")
def test_landing_runway() -> None:
    # TODO refactor/rethink the returned type
    assert belevingsvlucht.guess_landing_runway().name == "06"
    assert airbus_tree.guess_landing_runway().name == "23"


def test_douglas_peucker() -> None:
    # https://github.com/xoolive/traffic/pull/5
    x = [0, 100, 200]
    y = [0, 1, 0]
    z = [0, 0, 0]
    df3d = pd.DataFrame({"x": x, "y": y, "z": z})
    res = douglas_peucker(df=df3d, z="z", tolerance=1, z_factor=1)
    assert all(res)


def test_resample_unwrapped() -> None:
    # https://github.com/xoolive/traffic/issues/41

    df = pd.DataFrame.from_records(
        [
            (pd.Timestamp("2019-01-01 12:00:00"), 345),
            (pd.Timestamp("2019-01-01 12:00:30"), 355),
            (pd.Timestamp("2019-01-01 12:01:00"), 5),
            (pd.Timestamp("2019-01-01 12:01:30"), 15),
        ],
        columns=["timestamp", "track"],
    )

    resampled = Flight(df).resample("1s")
    assert len(resampled.query("50 < track < 300")) == 0

    resampled_10 = Flight(df).resample(10)
    assert len(resampled_10) == 10


def test_agg_time() -> None:
    flight = belevingsvlucht

    agg = flight.agg_time(groundspeed="mean", altitude="max")

    assert agg.max("groundspeed_mean") <= agg.max("groundspeed")
    assert agg.max("altitude_max") <= agg.max("altitude")


def test_comet() -> None:
    flight = belevingsvlucht

    takeoff = next(flight.query("altitude < 300").split("10T"))
    comet = takeoff.comet(minutes=1)

    assert takeoff.point.altitude + 2000 < comet.point.altitude  # type: ignore
