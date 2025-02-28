import sys
from operator import itemgetter
from typing import Any, cast

import pytest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from traffic.algorithms.douglas_peucker import douglas_peucker
from traffic.core import Flight
from traffic.data import airports, eurofirs, navaids
from traffic.data.samples import (
    airbus_tree,
    belevingsvlucht,
    calibration,
    featured,
    get_sample,
    zurich_airport,
)

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


@pytest.mark.slow
def test_subtract() -> None:
    flight = belevingsvlucht
    assert sum(1 for _ in flight.holding_pattern()) == 1
    hp = flight.next("holding_pattern")
    assert hp is not None
    assert pd.Timedelta("9 min") < hp.duration < pd.Timedelta("11 min")

    without_hp = flight - hp
    assert sum(1 for _ in without_hp) == 2
    total_duration = sum(
        (segment.duration for segment in without_hp), hp.duration
    )
    assert flight.duration - pd.Timedelta("1 min") <= total_duration
    assert total_duration <= flight.duration


def test_time_methods() -> None:
    flight = belevingsvlucht
    first10 = flight.first(minutes=10)
    last10 = flight.last(minutes=10)
    assert first10 is not None
    assert last10 is not None
    assert f"{first10.stop}" == "2018-05-30 15:31:37+00:00"
    assert f"{last10.start}" == "2018-05-30 20:12:57+00:00"

    first10 = flight.first("10 min")
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
    shorter = low.split("10 min").max()
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
            for chunk in subset.split("1 min")
            if chunk.duration > pd.Timedelta("5 minutes")
        )
        == 7
    )


@pytest.mark.slow
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

    clip_dk = airbus_tree.clip(eurofirs["EKDK"])
    assert clip_dk is not None
    assert clip_dk.duration < flight.duration

    clip_gg = airbus_tree.clip(eurofirs["EDGG"])
    assert clip_gg is not None
    assert clip_gg.duration < flight.duration

    clip_mm = airbus_tree.clip(eurofirs["EDMM"])
    assert clip_mm is not None
    assert clip_mm.duration < flight.duration


@pytest.mark.slow
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


def test_getattr() -> None:
    assert belevingsvlucht.vertical_rate_min < -3000
    assert 15000 < belevingsvlucht.altitude_max < 20000

    with pytest.raises(AttributeError, match="has no attribute"):
        belevingsvlucht.foo
    with pytest.raises(AttributeError, match="has no attribute"):
        belevingsvlucht.altitude_foo


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
        freq="30 min",
        factor=lambda f: f.distance() / f.cumulative_distance().max("cumdist"),
    )
    assert app.min("factor") < 1 / 15


def test_predict() -> None:
    flight = belevingsvlucht

    subset = flight.query("altitude < 300")
    assert subset is not None
    takeoff = subset.split("10 min").next()
    assert takeoff is not None
    forward = takeoff.predict(minutes=1)

    t_point = takeoff.point
    c_point = forward.point
    assert t_point is not None
    assert c_point is not None
    assert t_point.altitude + 2000 < c_point.altitude


def test_cumulative_distance() -> None:
    # https://github.com/xoolive/traffic/issues/61

    first = belevingsvlucht.first("30 min")
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

    cols = belevingsvlucht.agg_time(
        "5 min", altitude=("max", "mean")
    ).data.columns
    assert list(cols)[-3:] == ["rounded", "altitude_max", "altitude_mean"]

    cols = belevingsvlucht.agg_time(
        "5 min", altitude=lambda x: x.sum()
    ).data.columns
    assert list(cols)[-3:] == ["altitude", "rounded", "altitude_<lambda>"]

    def shh(x: Any) -> Any:
        return x.sum()

    cols = belevingsvlucht.agg_time("5 min", altitude=shh).data.columns
    assert list(cols)[-2:] == ["rounded", "altitude_shh"]


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
        belevingsvlucht.landing("EHLE").map(lambda f: f.resample("10s")).all()
    )
    assert result is not None
    assert 140 <= len(result) <= 160
