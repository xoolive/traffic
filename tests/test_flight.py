# fmt: off

import zipfile
from typing import Optional, cast

import pandas as pd
import pytest

from traffic.algorithms.douglas_peucker import douglas_peucker
from traffic.core import Flight, Traffic
from traffic.data import eurofirs, navaids, runways
from traffic.data.samples import (
    airbus_tree, belevingsvlucht, calibration, elal747, featured, get_sample
)

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
    last_20min = flight.last(minutes=20)
    assert last_20min is not None
    assert last_20min.mean("vertical_rate") < -500
    assert f"{flight.start}" == "2018-05-30 15:21:38+00:00"
    assert f"{flight.stop}" == "2018-05-30 20:22:56+00:00"
    assert flight.callsign == "TRA051"
    assert flight.title == "TRA051"
    flight2 = flight.assign(number="FAKE", flight_id="belevingsvlucht")
    assert flight2.title == "TRA051 – FAKE (belevingsvlucht)"
    assert flight.icao24 == "484506"
    assert flight.registration == "PH-HZO"
    assert flight.typecode == "B738"
    assert flight.aircraft == "484506 · 🇳🇱 PH-HZO (B738)"
    assert flight.flight_id is None


@pytest.mark.skipif(True, reason="TODO this is wrong...")
def test_get_traffic() -> None:
    traffic: Traffic = get_sample(featured, "traffic")
    assert "belevingsvlucht" in traffic.flight_ids


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
    first10 = flight.first(minutes=10)
    last10 = flight.last(minutes=10)
    assert first10 is not None
    assert last10 is not None
    assert f"{first10.stop}" == "2018-05-30 15:31:37+00:00"
    assert f"{last10.start}" == "2018-05-30 20:12:57+00:00"

    first10 = flight.first("10T")
    last10 = flight.last("10 minutes")
    assert first10 is not None
    assert last10 is not None
    assert f"{first10.stop}" == "2018-05-30 15:31:37+00:00"
    assert f"{last10.start}" == "2018-05-30 20:12:57+00:00"

    # between is a combination of before and after
    before_after = flight.before("2018-05-30 19:00")
    assert before_after is not None
    before_after = before_after.after("2018-05-30 18:00")
    between = flight.between("2018-05-30 18:00", "2018-05-30 19:00")

    # flight comparison made by distance computation
    assert before_after.distance(between).lateral.sum() < 1e-6  # type: ignore
    assert between.distance(before_after).vertical.sum() < 1e-6  # type: ignore

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

    assert a is not None and b is not None
    assert a.shorter_than(flight.duration)
    assert b.shorter_than(flight.duration)

    low = flight.query("altitude < 100")
    assert low is not None
    shorter = low.split("10T").max()
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
    ext_navaids = navaids.extent(ajaccio)
    assert ext_navaids is not None
    vor = ext_navaids["AJO"]
    assert vor is not None
    subset = ajaccio.bearing(vor).query("bearing.diff().abs() < .01")
    assert subset is not None
    assert (
        sum(
            1
            for chunk in subset.split("1T")
            if chunk.duration > pd.Timedelta("5 minutes")
        )
        == 7
    )


def test_geometry() -> None:
    flight: Flight = get_sample(featured, "belevingsvlucht")

    assert flight.distance() < 5  # returns to origin

    xy_length = flight.project_shape().length / 1852  # in nm
    last_pos = flight.cumulative_distance().at()
    assert last_pos is not None
    cumdist = last_pos.cumdist
    assert abs(xy_length - cumdist) / xy_length < 1e-3

    simplified = cast(Flight, flight.simplify(1e3))
    assert len(simplified) < len(flight)
    xy_length_s = simplified.project_shape().length / 1852
    assert xy_length_s < xy_length

    simplified_3d = flight.simplify(1e3, altitude="altitude")
    assert len(simplified) < len(simplified_3d) < len(flight)

    assert flight.intersects(eurofirs["EHAA"])
    assert flight.intersects(eurofirs["EHAA"].flatten())
    assert not flight.intersects(eurofirs["LFBB"])

    assert flight.distance(eurofirs["EHAA"]).data.distance.mean() < 0

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


def test_closest_point() -> None:
    from traffic.data import airports, navaids

    item = cast(
        Flight, belevingsvlucht.between("2018-05-30 16:00", "2018-05-30 17:00")
    ).closest_point(
        [
            airports["EHLE"],  # type: ignore
            airports["EHAM"],  # type: ignore
            navaids["NARAK"],  # type: ignore
        ]
    )
    res = f"{item.timestamp:%H:%M:%S}, {item.point}, {item.distance:.2f}m"
    assert res == "16:53:46, Lelystad Airport, 49.11m"


def test_landing_airport() -> None:
    assert belevingsvlucht.landing_airport().icao == "EHAM"
    assert airbus_tree.landing_airport().icao == "EDHI"


def test_landing_runway() -> None:
    segment = belevingsvlucht.last(minutes=30).on_runway("EHAM")  # type: ignore
    assert segment is not None
    assert segment.mean("altitude") < 0


def test_aligned_runway() -> None:
    assert belevingsvlucht.aligned_on_runway("EHAM").sum() == 2


@pytest.mark.skipif(skip_runways, reason="no runways")
def test_landing_ils() -> None:
    aligned: Optional["Flight"] = belevingsvlucht.aligned_on_ils(
        "EHAM"
    ).next()  # noqa: B305
    assert aligned is not None
    assert aligned.max("ILS") == "06"

    aligned = airbus_tree.aligned_on_ils("EDHI").next()  # noqa: B305
    assert aligned is not None
    assert aligned.max("ILS") == "23"


@pytest.mark.skipif(skip_runways, reason="no runways")
def test_takeoff_runway() -> None:
    # There are as many take-off as landing at EHLE
    nb_takeoff = sum(
        1
        for _ in belevingsvlucht.takeoff_from_runway("EHLE", threshold_alt=3000)
    )
    nb_landing = sum(1 for f in belevingsvlucht.aligned_on_ils("EHLE"))
    # with go-arounds, sometimes it just doesn't fit
    assert nb_takeoff <= nb_landing
    for aligned in belevingsvlucht.aligned_on_ils("EHLE"):
        after = belevingsvlucht.after(aligned.stop)
        assert after is not None
        takeoff = after.takeoff_from_runway("EHLE", threshold_alt=3000).next()
        # If a landing is followed by a take-off, then it's on the same runway
        assert takeoff is None or aligned.max("ILS") == takeoff.max("runway")


@pytest.mark.skipif(True, reason="only for local debug")
def test_takeoff_goaround() -> None:
    from traffic.data.datasets import landing_zurich_2019

    go_arounds = landing_zurich_2019.has("go_around").eval(
        desc="go_around", max_workers=8
    )

    for flight in go_arounds:
        for segment in flight.go_around():
            aligned = segment.aligned_on_ils("LSZH").next()
            takeoff = (
                segment.after(aligned.stop)
                .takeoff_from_runway("LSZH", threshold_alt=5000)
                .next()
            )
            assert (
                takeoff is None
                or takeoff.shorter_than("30s")
                or aligned.max("ILS") == takeoff.max("runway")
            )


def test_getattr() -> None:
    assert belevingsvlucht.vertical_rate_min < -3000
    assert 15000 < belevingsvlucht.altitude_max < 20000

    with pytest.raises(AttributeError, match="has no attribute"):
        belevingsvlucht.foo
    with pytest.raises(AttributeError, match="has no attribute"):
        belevingsvlucht.altitude_foo


@pytest.mark.skipif(skip_runways, reason="no runways")
def test_goaround() -> None:
    assert belevingsvlucht.go_around().next() is None  # noqa: B305
    assert belevingsvlucht.go_around("EHLE").sum() == 5

    # from traffic.data.datasets import landing_zurich_2019
    # assert sum(1 for _ in landing_zurich_2019["EWG7ME_1079"].go_around()) == 2

    # def many_goarounds(f):
    #     return sum(1 for _ in f.go_around()) > 1

    # gogo = (
    #     landing_zurich_2019.query("not simple")
    #     .iterate_lazy()
    #     .pipe(many_goarounds)
    #     .eval(desc="", max_workers=8)
    # )

    # assert gogo.flight_ids == {"SWR287A_10099", "EWG7ME_1079"}


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
            (pd.Timestamp("2019-01-01 12:00:00Z"), 345),
            (pd.Timestamp("2019-01-01 12:00:30Z"), 355),
            (pd.Timestamp("2019-01-01 12:01:00Z"), 5),
            (pd.Timestamp("2019-01-01 12:01:30Z"), 15),
        ],
        columns=["timestamp", "track"],
    )

    resampled = Flight(df).resample("1s")
    assert resampled.query("50 < track < 300") is None

    resampled_10 = Flight(df).resample(10)
    assert len(resampled_10) == 10


def test_agg_time() -> None:
    flight = belevingsvlucht

    agg = flight.agg_time(groundspeed="mean", altitude="max")

    assert agg.max("groundspeed_mean") <= agg.max("groundspeed")
    assert agg.max("altitude_max") <= agg.max("altitude")

    app = flight.resample("30s").apply_time(
        freq="30T",
        factor=lambda f: f.distance() / f.cumulative_distance().max("cumdist"),
    )
    assert app.min("factor") < 1 / 15


def test_comet() -> None:
    flight = belevingsvlucht

    subset = flight.query("altitude < 300")
    assert subset is not None
    takeoff = subset.split("10T").next()  # noqa: B305
    assert takeoff is not None
    comet = takeoff.comet(minutes=1)

    t_point = takeoff.point
    c_point = comet.point
    assert t_point is not None
    assert c_point is not None
    assert t_point.altitude + 2000 < c_point.altitude


def test_cumulative_distance() -> None:
    # https://github.com/xoolive/traffic/issues/61

    f1 = (
        belevingsvlucht.before("2018-05-30 20:17:58")  # type: ignore
        .last(minutes=15)
        .cumulative_distance(compute_track=True)
        .last(minutes=10)
        .filter(compute_gs=17)
        .filter(compute_gs=53)
        .filter(compute_track=17)
    )

    f2 = (
        belevingsvlucht.before("2018-05-30 20:17:58")  # type: ignore
        .last(minutes=15)
        .cumulative_distance(compute_track=True, reverse=True)
        .last(minutes=10)
        .filter(compute_gs=17)
        .filter(compute_gs=53)
        .filter(compute_track=17)
    )

    assert f1.diff(["cumdist"]).mean("cumdist_diff") > 0
    assert f2.diff(["cumdist"]).mean("cumdist_diff") < 0

    assert abs(f1.diff(["compute_gs"]).mean("compute_gs_diff")) < 1
    assert abs(f2.diff(["compute_gs"]).mean("compute_gs_diff")) < 1

    assert abs(f1.diff(["compute_track"]).mean("compute_track_diff")) < 1
    assert abs(f2.diff(["compute_track"]).mean("compute_track_diff")) < 1

    # check that first value is non-zero
    assert f1.data.iloc[0].compute_track > 1
    assert f1.data.iloc[0].compute_gs > 1


def test_agg_time_colnames() -> None:
    # https://github.com/xoolive/traffic/issues/66

    cols = belevingsvlucht.agg_time("5T", altitude=("max", "mean")).data.columns
    assert list(cols)[-3:] == ["rounded", "altitude_max", "altitude_mean"]

    cols = belevingsvlucht.agg_time(
        "5T", altitude=lambda x: x.sum()
    ).data.columns
    assert list(cols)[-3:] == ["altitude", "rounded", "altitude_<lambda>"]

    def shh(x):
        x.sum()

    cols = belevingsvlucht.agg_time("5T", altitude=shh).data.columns
    assert list(cols)[-2:] == ["rounded", "altitude_shh"]


def test_parking_position() -> None:
    pp = elal747.parking_position("LIRF")
    assert pp is not None
    assert pp.max("parking_position") == "702"
