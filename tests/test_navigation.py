import zipfile
from typing import Optional

import httpx
import pytest

import pandas as pd
from traffic.core import Flight
from traffic.data import airports, runways
from traffic.data.samples import (
    airbus_tree,
    belevingsvlucht,
    elal747,
    landing_denver,
    zurich_airport,
)

# This part only serves on travis when the downloaded file is corrupted
# This shouldn't happen much as caching is now activated.
skip_runways = False

try:
    _ = runways.runways
except zipfile.BadZipFile:
    skip_runways = True


def test_landing_airport() -> None:  # TODO
    assert belevingsvlucht.infer_airport("landing").icao == "EHAM"
    assert airbus_tree.infer_airport("landing").icao == "EDHI"


@pytest.mark.skipif(skip_runways, reason="no runways")
def test_landing_based_on_ils() -> None:
    from traffic.algorithms.navigation.landing import LandingAlignedOnILS

    # by default
    aligned: Optional["Flight"] = belevingsvlucht.landing("EHAM").next()
    assert aligned is not None
    assert aligned.max("ILS") == "06"

    # inside an aggregation method like next, max, final, etc.
    aligned = belevingsvlucht.final("aligned('EHLE', method='ils')")
    assert aligned is not None
    assert aligned.ILS_max == "23"

    # with a specific method
    landing_method = LandingAlignedOnILS("EDHI")
    aligned = airbus_tree.landing(method=landing_method).next()
    assert aligned is not None
    assert aligned.max("ILS") == "23"

    # corner case with high elevation airport
    segment = landing_denver.landing("KDEN").next()
    assert segment is not None
    assert segment.ILS_max == "26"
    assert segment.data.altitude.min() == 5575


@pytest.mark.slow
def test_infer_flightplan() -> None:
    from traffic.data.samples import switzerland

    df = switzerland["BAW585E"].infer_flightplan()
    assert df is not None
    assert df.navaid.to_list() == ["RESIA", "LUL", "IXILU"]

    df = switzerland["EZY24DP"].infer_flightplan()
    assert df is not None
    assert df.navaid.to_list() == ["RESIA", "ZH367", "LUL"]


def test_takeoff() -> None:
    for aligned in belevingsvlucht.landing("EHLE"):
        after = belevingsvlucht.after(aligned.stop)
        assert after is not None
        takeoff = after.takeoff("EHLE", max_ft_above_airport=3000).next()
        # If a landing is followed by a take-off, then it's on the same runway
        assert takeoff is None or aligned.max("ILS") == takeoff.max("runway")

    # Flight SWR5220 - Should take off from runway 28 at LSZH
    flight_swr5220 = zurich_airport["SWR5220"]
    segment = flight_swr5220.takeoff("LSZH", method="track_based").next()
    assert segment is not None
    assert segment.runway_max == "28", (
        f"SWR5220 should take off from runway 28, got {segment}"
    )

    # Flight VJT796 - Should return None since it's a landing at LSZH
    flight_vjt796 = zurich_airport["VJT796"]
    segment = flight_vjt796.takeoff("LSZH", method="track_based").next()
    assert segment is None, f"VJT796 should return None, got {segment}"

    # Flight ACA879 - Should take off from runway 16 at LSZH
    flight_aca879 = zurich_airport["ACA879"]
    segment = flight_aca879.takeoff("LSZH", method="track_based").next()
    assert segment is not None
    assert segment.runway_max == "16", (
        f"ACA879 should take off from runway 16, got {segment}"
    )

    # Flight SWR5220 tested at LFPG - Should return None
    segment = flight_swr5220.takeoff("LFPG", method="track_based").next()
    assert segment is None, (
        f"SWR5220 should return None for LFPG, got {segment}"
    )


@pytest.mark.skipif(skip_runways, reason="no runways")
@pytest.mark.slow
def test_takeoff_runway() -> None:
    from traffic.algorithms.navigation.takeoff import (
        PolygonBasedRunwayDetection,
    )

    takeoff = PolygonBasedRunwayDetection("EHLE", max_ft_above_airport=3000)
    # There are as many take-off as landing at EHLE
    nb_takeoff = sum(1 for _ in belevingsvlucht.takeoff(method=takeoff))
    nb_landing = sum(1 for f in belevingsvlucht.landing("EHLE"))
    # with go-arounds, sometimes it just doesn't fit
    assert nb_takeoff <= nb_landing
    for aligned in belevingsvlucht.landing("EHLE"):
        after = belevingsvlucht.after(aligned.stop)
        assert after is not None
        segment = after.takeoff(method=takeoff).next()
        # If a landing is followed by a take-off, then it's on the same runway
        assert segment is None or aligned.max("ILS") == segment.max("runway")


@pytest.mark.slow
def test_many_goarounds() -> None:
    from traffic.data.datasets import landing_zurich_2019

    def many_goarounds(flight: Flight) -> bool:
        return sum(1 for _ in flight.go_around()) > 1

    result = (
        landing_zurich_2019[["SWR287A_10099", "EWG7ME_1079"]]
        .iterate_lazy()
        .pipe(many_goarounds)
        .eval()
    )
    assert sorted(result.flight_ids) == ["EWG7ME_1079", "SWR287A_10099"]


@pytest.mark.xfail(
    raises=httpx.TransportError, reason="Quotas on OpenStreetMap"
)
def test_parking_position() -> None:
    lszh_pp = (
        airports["LSZH"]._openstreetmap().query('aeroway == "parking_position"')
    )
    lirf_pp = (
        airports["LIRF"]._openstreetmap().query('aeroway == "parking_position"')
    )

    pp = elal747.parking_position("LIRF", parking_positions=lirf_pp).next()
    assert pp is not None
    assert pp.max("parking_position") == "702"

    # Landing aircraft
    flight = zurich_airport["EDW229"]
    assert flight is not None

    pp = flight.parking_position(
        "LSZH", parking_positions=lszh_pp, buffer_size=1e-4
    ).next()
    assert pp is not None
    assert 5 < pp.duration.total_seconds() < 10
    assert pp.parking_position_max == "A49"


# def test_slow_taxi() -> None:  # TODO
#     flight = zurich_airport["SWR137H"]
#     assert flight is not None
#     slow_durations = sum(
#         ((slow_segment.duration,) for slow_segment in flight.slow_taxi()),
#         (),
#     )
#
#     assert len(slow_durations) == 4
#     slow_durations_sum = sum(slow_durations, pd.Timedelta(0))
#     assert slow_durations_sum > pd.Timedelta("0 days 00:06:00")
#
#     flight = zurich_airport["ACA879"]
#     assert flight is not None
#     assert flight.slow_taxi().next() is None


@pytest.mark.xfail(
    raises=httpx.TransportError, reason="Quotas on OpenStreetMap"
)
def test_pushback_parking_position() -> None:
    lszh_pp = (
        airports["LSZH"]._openstreetmap().query('aeroway == "parking_position"')
    )

    flight = zurich_airport["AEE5ZH"]
    assert flight is not None
    parking_position = flight.parking_position(
        "LSZH", parking_positions=lszh_pp
    ).max()
    pushback = flight.pushback(
        "LSZH", method="parking_position", parking_positions=lszh_pp
    )

    assert parking_position is not None
    assert pushback is not None
    assert (
        parking_position.parking_position_max
        == pushback.parking_position_max
        == "A10"
    )
    assert pushback.start <= parking_position.stop
    assert pushback.stop >= parking_position.stop


def test_pushback_parking_area() -> None:
    flight = zurich_airport["AEE5ZH"]
    assert flight is not None
    parking_position = flight.pushback("LSZH", method="parking_area")
    assert parking_position is not None
    assert parking_position.duration > pd.Timedelta("1 min")


# @pytest.mark.xfail(
#     raises=httpx.TransportError, reason="Quotas on OpenStreetMap"
# )
# @pytest.mark.slow
# def test_on_taxiway() -> None:  # TODO
#     lszh_taxiway = (
#         airports["LSZH"]._openstreetmap().query('aeroway == "taxiway"')
#     )
#     lirf_taxiway = (
#         airports["LIRF"]._openstreetmap().query('aeroway == "taxiway"')
#     )
#     llbg_taxiway = (
#         airports["LLBG"]._openstreetmap().query('aeroway == "taxiway"')
#     )
#
#     flight = zurich_airport["ACA879"]
#     assert flight is not None
#     assert len(flight.on_taxiway(lszh_taxiway)) == 3
#
#     last_stop = pd.Timestamp(0, tz="utc")
#     twy_names = ["Charlie", "Echo", "E2"]
#     for i, twy_seg in enumerate(flight.on_taxiway(lszh_taxiway)):
#         assert twy_seg.start >= last_stop
#         last_stop = twy_seg.stop
#         assert twy_seg.taxiway_max == twy_names[i]
#
#     flight = zurich_airport["SWISS"]
#     assert flight is not None
#     assert flight.on_taxiway(lszh_taxiway).next() is None
#
#     # another airport
#     last_stop = pd.Timestamp(0, tz="utc")
#     twy_names = ["V", "Z", "M", "R", "B", "BB"]
#     flight = elal747
#     assert flight is not None
#     for i, twy_seg in enumerate(flight.on_taxiway(lirf_taxiway)):
#         assert twy_seg.start >= last_stop
#         last_stop = twy_seg.stop
#         assert twy_seg.taxiway_max == twy_names[i]
#
#     # Landing
#     last_stop = pd.Timestamp(0, tz="utc")
#     twy_names = ["K", "M1", "D6"]
#     for i, twy_seg in enumerate(flight.on_taxiway(llbg_taxiway)):
#         assert twy_seg.start >= last_stop
#         last_stop = twy_seg.stop
#         assert twy_seg.taxiway_max == twy_names[i]
#
#     # Flight that leaves and come back to the airport
#     flight = zurich_airport["SWR5220"]
#     assert flight is not None
#     assert len(flight.on_taxiway(lszh_taxiway)) == 9
#

# @pytest.mark.slow
# def test_ground_trajectory() -> None:  # TODO
#     flight_iterate = belevingsvlucht.ground_trajectory("EHAM")
#     takeoff = next(flight_iterate)
#     assert (
#         pd.Timestamp("2018-05-30 15:21:00+00:00")
#         < takeoff.stop
#         < pd.Timestamp("2018-05-30 15:22:00+00:00")
#     )
#     landing = next(flight_iterate)
#     assert (
#         pd.Timestamp("2018-05-30 20:17:00+00:00")
#         < landing.start
#         < pd.Timestamp("2018-05-30 20:18:00+00:00")
#     )
#     assert next(flight_iterate, None) is None
#
#     assert belevingsvlucht.ground_trajectory("EHLE").sum() == 0
#
#     flight = zurich_airport["SWR5220"]
#     assert flight is not None
#
#     flight_iterate = flight.ground_trajectory("LSZH")
#     takeoff = next(flight_iterate)
#     assert (
#         pd.Timestamp("2019-11-05 13:05:00+00:00")
#         < takeoff.stop
#         < pd.Timestamp("2019-11-05 13:06:00+00:00")
#     )
#     landing = next(flight_iterate)
#     assert (
#         pd.Timestamp("2019-11-05 16:36:00+00:00")
#         < landing.start
#         < pd.Timestamp("2019-11-05 16:37:00+00:00")
#     )
#     assert next(flight_iterate, None) is None


# @pytest.mark.skipif(version > (3, 13), reason="onnxruntime not ready")
@pytest.mark.slow
def test_holding_pattern() -> None:
    holding_pattern = belevingsvlucht.holding_pattern().next()
    assert holding_pattern is not None
    assert (
        holding_pattern.between("2018-05-30 15:45", "2018-05-30 15:50")
        is not None
    )


# @pytest.mark.skipif(version > (3, 13), reason="onnxruntime not ready")
@pytest.mark.slow
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
        "landing('LSZH')",
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
