import json
import sys
import zipfile
from operator import itemgetter
from pathlib import Path
from typing import Any, Optional, cast

import httpx
import pytest
from cartes.osm import Overpass

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from traffic.algorithms.douglas_peucker import douglas_peucker
from traffic.core import Flight, Traffic
from traffic.data import airports, eurofirs, navaids, runways
from traffic.data.samples import (
    airbus_tree,
    belevingsvlucht,
    calibration,
    elal747,
    featured,
    get_sample,
    zurich_airport,
)

# This part only serves on travis when the downloaded file is corrupted
# This shouldn't happen much as caching is now activated.
skip_runways = False

try:
    _ = runways.runways
except zipfile.BadZipFile:
    skip_runways = True

data_folder = Path(__file__).parent / "json"
lszh_json = json.loads((data_folder / "overpass_lszh.json").read_text())
lirf_json = json.loads((data_folder / "overpass_lirf.json").read_text())
llbg_json = json.loads((data_folder / "overpass_llbg.json").read_text())

version = sys.version_info


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
    assert flight2.title == "TRA051 – FAKE (belevingsvlucht)"  # noqa: RUF001
    assert flight.icao24 == "484506"
    assert flight.registration == "PH-HZO"
    assert flight.typecode == "B738"
    assert (
        repr(flight.aircraft) == "Tail(icao24='484506', registration='PH-HZO',"
        " typecode='B738', flag='🇳🇱')"
    )
    assert flight.flight_id is None


def test_dtype() -> None:
    # See PR 324

    assert any(
        dtype == np.float64
        for dtype in belevingsvlucht.resample("1s").data.dtypes
    )


@pytest.mark.skipif(True, reason="TODO this is wrong...")
def test_get_traffic() -> None:
    traffic = cast(Traffic, get_sample(featured, "traffic"))
    assert "belevingsvlucht" in traffic.flight_ids


def test_iterators() -> None:
    flight = belevingsvlucht
    assert min(flight.timestamp) == flight.start
    assert max(flight.timestamp) == flight.stop
    assert min(flight.coords)[0] == flight.min("longitude")
    assert max(flight.coords)[0] == flight.max("longitude")

    max_time = max(flight.coords4d(), key=itemgetter("timestamp"))
    last_point = flight.at()
    assert last_point is not None
    assert max_time["longitude"] == last_point.longitude
    assert max_time["latitude"] == last_point.latitude
    assert max_time["altitude"] == last_point.altitude

    max_xy_time = list(flight.xy_time)[-1]
    assert max_xy_time[0] == last_point.longitude
    assert max_xy_time[1] == last_point.latitude
    assert max_xy_time[2] == last_point.timestamp.to_pydatetime().timestamp()


def test_subtract() -> None:
    flight = belevingsvlucht
    assert sum(1 for _ in flight.holding_pattern()) == 1
    hp = flight.next("holding_pattern")
    assert hp is not None
    assert pd.Timedelta("9T") < hp.duration < pd.Timedelta("11T")

    without_hp = flight - hp
    assert sum(1 for _ in without_hp) == 2
    total_duration = sum(
        (segment.duration for segment in without_hp), hp.duration
    )
    assert flight.duration - pd.Timedelta("1T") <= total_duration
    assert total_duration <= flight.duration


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
    ajaccio = cast(Flight, get_sample(calibration, "ajaccio"))
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
    flight = cast(Flight, get_sample(featured, "belevingsvlucht"))

    assert flight.distance() < 5  # returns to origin

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

    EHAA = eurofirs["EHAA"]
    LFBB = eurofirs["LFBB"]
    assert EHAA is not None
    assert LFBB is not None

    assert flight.intersects(EHAA)
    assert flight.intersects(EHAA.flatten())
    assert not flight.intersects(LFBB)

    assert flight.distance(EHAA).data.distance.mean() < 0

    airbus_tree = cast(Flight, get_sample(featured, "airbus_tree"))
    clip_dk = airbus_tree.clip(eurofirs["EKDK"])
    assert clip_dk is not None
    assert clip_dk.duration < flight.duration

    clip_gg = airbus_tree.clip(eurofirs["EDGG"])
    assert clip_gg is not None
    assert clip_gg.duration < flight.duration

    clip_mm = airbus_tree.clip(eurofirs["EDMM"])
    assert clip_mm is not None
    assert clip_mm.duration < flight.duration


def test_clip_iterate() -> None:
    schiphol = airports["EHAM"]
    assert schiphol is not None
    schiphol_shape = schiphol.shape
    assert schiphol_shape is not None
    flight_iterate = belevingsvlucht.clip_iterate(
        schiphol_shape.buffer(2e-3), strict=False
    )
    takeoff = next(flight_iterate)
    assert (
        pd.Timestamp("2018-05-30 15:21:00+00:00")
        < takeoff.stop
        < pd.Timestamp("2018-05-30 15:22:00+00:00")
    )
    landing = next(flight_iterate)
    assert (
        pd.Timestamp("2018-05-30 20:17:00+00:00")
        < landing.start
        < pd.Timestamp("2018-05-30 20:18:00+00:00")
    )


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

    lelystad = airports["EHLE"]
    assert lelystad is not None

    schiphol = airports["EHAM"]
    assert schiphol is not None

    narak = navaids["NARAK"]
    assert narak is not None

    item = cast(
        Flight, belevingsvlucht.between("2018-05-30 16:00", "2018-05-30 17:00")
    ).closest_point([lelystad, schiphol, narak])
    assert item.point == "Lelystad Airport"


def test_landing_airport() -> None:
    assert belevingsvlucht.landing_airport().icao == "EHAM"
    assert airbus_tree.landing_airport().icao == "EDHI"


def test_landing_runway() -> None:
    last = belevingsvlucht.last(minutes=30)
    assert last is not None
    segment = last.aligned_on_runway("EHAM").max()
    assert segment is not None
    assert segment.mean("altitude") < 0


def test_aligned_runway() -> None:
    assert belevingsvlucht.aligned_on_runway("EHAM").sum() == 2


@pytest.mark.skipif(skip_runways, reason="no runways")
def test_landing_ils() -> None:
    aligned: Optional["Flight"] = belevingsvlucht.aligned_on_ils("EHAM").next()
    assert aligned is not None
    assert aligned.max("ILS") == "06"

    aligned = belevingsvlucht.final("aligned_on_ils('EHLE')")
    assert aligned is not None
    assert aligned.ILS_max == "23"

    aligned = airbus_tree.aligned_on_ils("EDHI").next()
    assert aligned is not None
    assert aligned.max("ILS") == "23"


def test_compute_navpoints() -> None:
    from traffic.data.samples import switzerland

    df = switzerland["BAW585E"].compute_navpoints()
    assert df is not None
    assert df.navaid.to_list() == ["RESIA", "LUL", "IXILU"]

    df = switzerland["EZY24DP"].compute_navpoints()
    assert df is not None
    assert df.navaid.to_list() == ["RESIA", "ZH367", "LUL"]


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
    assert go_arounds is not None

    for flight in go_arounds:
        for segment in flight.go_around():
            aligned = segment.aligned_on_ils("LSZH").next()
            assert aligned is not None
            takeoff = (
                cast(Flight, segment.after(aligned.stop))
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
    assert belevingsvlucht.go_around().next() is None
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


def test_resample_how_argument() -> None:
    df = pd.DataFrame.from_records(
        [
            (pd.Timestamp("2019-01-01 12:00:00Z"), 30000, 0),
            (pd.Timestamp("2019-01-01 12:00:05Z"), 25000, 5),
            (pd.Timestamp("2019-01-01 12:00:10Z"), 27000, 10),
        ],
        columns=["timestamp", "altitude", "fake"],
    )

    fake_interpolate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fake_ffill = [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10]
    fake_nothing = [
        0,
        None,
        None,
        None,
        None,
        5,
        None,
        None,
        None,
        None,
        10,
    ]
    altitude_interpolate = [
        30000.0,
        29000.0,
        28000.0,
        27000.0,
        26000.0,
        25000.0,
        25400.0,
        25800.0,
        26200.0,
        26600.0,
        27000.0,
    ]
    altitude_ffill = [
        30000.0,
        30000.0,
        30000.0,
        30000.0,
        30000.0,
        25000.0,
        25000.0,
        25000.0,
        25000.0,
        25000.0,
        27000.0,
    ]

    resampled_interpolate = Flight(df).resample("1s", how="interpolate")
    pd.testing.assert_frame_equal(
        resampled_interpolate.data[["altitude", "fake"]],
        pd.DataFrame(
            {"altitude": altitude_interpolate, "fake": fake_interpolate}
        ),
        check_dtype=False,
    )

    altitude_interpolate_quadratic = [
        30000.0,
        28440.0,
        27160.0,
        26160.0,
        25440.0,
        25000.0,
        24840.0,
        24960.0,
        25360.0,
        26040.0,
        27000.0,
    ]
    resampled_interpolate_quadratic = Flight(df).resample(
        "1s",
        how="interpolate",
        interpolate_kw={"method": "polynomial", "order": 2},
    )
    pd.testing.assert_frame_equal(
        resampled_interpolate_quadratic.data[["altitude", "fake"]],
        pd.DataFrame(
            {
                "altitude": altitude_interpolate_quadratic,
                "fake": fake_interpolate,
            }
        ),
        check_dtype=False,
        check_exact=False,  # accomodate for small rounding errors
    )

    resampled_ffill = Flight(df).resample("1s", how="ffill")
    pd.testing.assert_frame_equal(
        resampled_ffill.data[["altitude", "fake"]],
        pd.DataFrame({"altitude": altitude_ffill, "fake": fake_ffill}),
        check_dtype=False,
    )

    resampled_mixed = Flight(df).resample(
        "1s", how=dict(interpolate=["altitude"], ffill=["fake"])
    )
    pd.testing.assert_frame_equal(
        resampled_mixed.data[["altitude", "fake"]],
        pd.DataFrame({"altitude": altitude_interpolate, "fake": fake_ffill}),
        check_dtype=False,
    )

    resampled_partial = Flight(df).resample("1s", how=dict(ffill=["altitude"]))
    pd.testing.assert_frame_equal(
        resampled_partial.data[["altitude", "fake"]],
        pd.DataFrame({"altitude": altitude_ffill, "fake": fake_nothing}),
        check_dtype=False,
    )


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


def test_resample_projection() -> None:
    flight = Flight(
        pd.DataFrame.from_dict(
            dict(
                timestamp=[
                    pd.Timestamp("2018-01-01 04:09:29Z"),
                    pd.Timestamp("2018-01-01 10:09:46Z"),
                ],
                latitude=[40.64, 49.0],
                longitude=[-73.81, 2.81],
            )
        )
    )

    r1 = flight.resample("1s").cumulative_distance()
    r2 = flight.resample("1s", projection="lcc").cumulative_distance()

    assert r1.cumdist_sum > r2.cumdist_sum


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


def test_forward() -> None:
    flight = belevingsvlucht

    subset = flight.query("altitude < 300")
    assert subset is not None
    takeoff = subset.split("10T").next()
    assert takeoff is not None
    forward = takeoff.forward(minutes=1)

    t_point = takeoff.point
    c_point = forward.point
    assert t_point is not None
    assert c_point is not None
    assert t_point.altitude + 2000 < c_point.altitude


def test_cumulative_distance() -> None:
    # https://github.com/xoolive/traffic/issues/61

    first = belevingsvlucht.first("30T")
    assert first is not None
    f1 = first.cumulative_distance()
    f2 = first.cumulative_distance(reverse=True)
    assert f1.max("cumdist") == f1.max("cumdist")  # bugfix #197

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

    def shh(x: Any) -> Any:
        return x.sum()

    cols = belevingsvlucht.agg_time("5T", altitude=shh).data.columns
    assert list(cols)[-2:] == ["rounded", "altitude_shh"]


@pytest.mark.xfail(
    raises=httpx.TransportError, reason="Quotas on OpenStreetMap"
)
def test_parking_position() -> None:
    # These tests with calls to OpenStreetMap are a bit tricky and subject to
    # uncontrolled edits on OpenStreetMap.
    lszh_pp = Overpass(lszh_json).query('aeroway == "parking_position"')
    lirf_pp = Overpass(lirf_json).query('aeroway == "parking_position"')

    pp = elal747.on_parking_position("LIRF", parking_positions=lirf_pp).next()
    assert pp is not None
    assert pp.max("parking_position") == "702"

    # Landing aircraft
    flight = zurich_airport["EDW229"]
    assert flight is not None

    pp = flight.on_parking_position(
        "LSZH", parking_positions=lszh_pp, buffer_size=1e-4
    ).next()
    assert pp is not None
    assert 5 < pp.duration.total_seconds() < 10
    assert pp.parking_position_max == "A49"


def test_from_inertial() -> None:
    flight = zurich_airport["ENT57BW"]
    assert flight is not None
    assert flight.is_from_inertial()

    flight = zurich_airport["CAI3208"]
    assert flight is not None
    assert not flight.is_from_inertial()


def test_slow_taxi() -> None:
    flight = zurich_airport["SWR137H"]
    assert flight is not None
    slow_durations = sum(
        ((slow_segment.duration,) for slow_segment in flight.slow_taxi()),
        (),
    )

    assert len(slow_durations) == 4
    slow_durations_sum = sum(slow_durations, pd.Timedelta(0))
    assert slow_durations_sum > pd.Timedelta("0 days 00:06:00")

    flight = zurich_airport["ACA879"]
    assert flight is not None
    assert flight.slow_taxi().next() is None


@pytest.mark.xfail(
    raises=httpx.TransportError, reason="Quotas on OpenStreetMap"
)
def test_pushback() -> None:
    # These tests with calls to OpenStreetMap are a bit tricky and subject to
    # uncontrolled edits on OpenStreetMap.
    lszh_pp = Overpass(lszh_json).query('aeroway == "parking_position"')

    flight = zurich_airport["AEE5ZH"]
    assert flight is not None
    parking_position = flight.on_parking_position(
        "LSZH", parking_positions=lszh_pp
    ).max()
    pushback = flight.pushback("LSZH", parking_positions=lszh_pp)

    assert parking_position is not None
    assert pushback is not None
    assert (
        parking_position.parking_position_max
        == pushback.parking_position_max
        == "A10"
    )
    assert pushback.start <= parking_position.stop
    assert pushback.stop >= parking_position.stop


@pytest.mark.xfail(
    raises=httpx.TransportError, reason="Quotas on OpenStreetMap"
)
def test_on_taxiway() -> None:
    # These tests with calls to OpenStreetMap are a bit tricky and subject to
    # uncontrolled edits on OpenStreetMap.
    lszh_taxiway = Overpass(lszh_json).query('aeroway == "taxiway"')
    lirf_taxiway = Overpass(lirf_json).query('aeroway == "taxiway"')
    llbg_taxiway = Overpass(llbg_json).query('aeroway == "taxiway"')

    flight = zurich_airport["ACA879"]
    assert flight is not None
    assert len(flight.on_taxiway(lszh_taxiway)) == 3

    last_stop = pd.Timestamp(0, tz="utc")
    twy_names = ["C", "E", "E2"]
    for i, twy_seg in enumerate(flight.on_taxiway(lszh_taxiway)):
        assert twy_seg.start >= last_stop
        last_stop = twy_seg.stop
        assert twy_seg.taxiway_max == twy_names[i]

    flight = zurich_airport["SWISS"]
    assert flight is not None
    assert flight.on_taxiway(lszh_taxiway).next() is None

    # another airport
    last_stop = pd.Timestamp(0, tz="utc")
    twy_names = ["V", "Z", "M", "R", "B", "BB"]
    flight = elal747
    assert flight is not None
    for i, twy_seg in enumerate(flight.on_taxiway(lirf_taxiway)):
        assert twy_seg.start >= last_stop
        last_stop = twy_seg.stop
        assert twy_seg.taxiway_max == twy_names[i]

    # Landing
    last_stop = pd.Timestamp(0, tz="utc")
    twy_names = ["K", "M1", "D6"]
    for i, twy_seg in enumerate(flight.on_taxiway(llbg_taxiway)):
        assert twy_seg.start >= last_stop
        last_stop = twy_seg.stop
        assert twy_seg.taxiway_max == twy_names[i]

    # Flight that leaves and come back to the airport
    flight = zurich_airport["SWR5220"]
    assert flight is not None
    assert len(flight.on_taxiway(lszh_taxiway)) == 8


def test_ground_trajectory() -> None:
    flight_iterate = belevingsvlucht.ground_trajectory("EHAM")
    takeoff = next(flight_iterate)
    assert (
        pd.Timestamp("2018-05-30 15:21:00+00:00")
        < takeoff.stop
        < pd.Timestamp("2018-05-30 15:22:00+00:00")
    )
    landing = next(flight_iterate)
    assert (
        pd.Timestamp("2018-05-30 20:17:00+00:00")
        < landing.start
        < pd.Timestamp("2018-05-30 20:18:00+00:00")
    )
    assert next(flight_iterate, None) is None

    assert belevingsvlucht.ground_trajectory("EHLE").sum() == 0

    flight = zurich_airport["SWR5220"]
    assert flight is not None

    flight_iterate = flight.ground_trajectory("LSZH")
    takeoff = next(flight_iterate)
    assert (
        pd.Timestamp("2019-11-05 13:05:00+00:00")
        < takeoff.stop
        < pd.Timestamp("2019-11-05 13:06:00+00:00")
    )
    landing = next(flight_iterate)
    assert (
        pd.Timestamp("2019-11-05 16:36:00+00:00")
        < landing.start
        < pd.Timestamp("2019-11-05 16:37:00+00:00")
    )
    assert next(flight_iterate, None) is None


def test_DME_NSE_computation() -> None:
    flight = zurich_airport["EDW229"]
    assert flight is not None

    dme_zone = navaids.query("type == 'DME'").extent(flight)  # type: ignore
    assert dme_zone is not None

    dmes = dme_zone.query('not description.str.endswith("ILS")')
    assert dmes is not None

    segment = flight.resample("60s").first("2min")
    assert segment is not None
    result_df = segment.compute_DME_NSE(dmes).data

    expected = pd.DataFrame(
        [[0.293, "TRA_ZUE"], [0.279, "KLO_TRA"]],
        columns=["NSE", "NSE_idx"],
    )

    assert_frame_equal(result_df[["NSE", "NSE_idx"]], expected, rtol=1e-3)


# @pytest.mark.skipif(version > (3, 13), reason="onnxruntime not ready")
def test_holding_pattern() -> None:
    holding_pattern = belevingsvlucht.holding_pattern().next()
    assert holding_pattern is not None
    assert (
        holding_pattern.between("2018-05-30 15:45", "2018-05-30 15:50")
        is not None
    )


# @pytest.mark.skipif(version > (3, 13), reason="onnxruntime not ready")
def test_label() -> None:
    from traffic.data.datasets import landing_zurich_2019

    labelled = belevingsvlucht.first("1h").label(
        "holding_pattern", holding=True
    )
    holding = labelled.query("holding")
    assert holding is not None
    assert holding.between("2018-05-30 15:45", "2018-05-30 15:50") is not None

    f = landing_zurich_2019["SWR287A_10099"]
    labelled = f.label(
        "aligned_on_ils('LSZH')",
        ILS="{segment.ILS_max}",
        touch=lambda segment: segment.stop,
        index=lambda i, segment: i,
        alt_min=lambda i, segment, flight: segment.altitude_min,
        duration="lambda segment: segment.stop - segment.start",
    )

    assert labelled.index_max == 2
    ils = set(ils for ils in labelled.ILS_unique if not pd.isna(ils))
    assert ils == {"14", "28"}
    min_duration = labelled.data.duration.astype("timedelta64[s]").min()
    assert min_duration > pd.Timedelta("2 min 30 s")


def test_split_condition() -> None:
    def no_split_below_5000(f1: Flight, f2: Flight) -> bool:
        return (  # type: ignore
            f1.data.iloc[-1].altitude >= 5000
            or f2.data.iloc[0].altitude >= 5000
        )

    f_max = (
        belevingsvlucht.query("altitude > 2000")  # type: ignore
        .split(
            "1 min",
            condition=no_split_below_5000,
        )
        .max()
    )

    assert f_max is not None
    assert f_max.start - belevingsvlucht.start < pd.Timedelta("5 min")
    assert belevingsvlucht.stop - f_max.stop < pd.Timedelta("10 min")


def test_split_map() -> None:
    result = (
        belevingsvlucht.aligned_on_ils("EHLE")
        .map(lambda f: f.resample("10s"))
        .all()
    )
    assert result is not None
    assert 140 <= len(result) <= 160
