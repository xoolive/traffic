import pandas as pd
from traffic.core import Flight
from traffic.data import eurofirs
from traffic.data.samples import switzerland


def test_properties() -> None:
    assert len(switzerland) == 1244
    assert f"{switzerland.start_time}" == "2018-08-01 05:00:00+00:00"
    assert f"{switzerland.end_time}" == "2018-08-01 21:59:50+00:00"

    assert len(switzerland.callsigns) == 1243
    assert len(switzerland.icao24) == 842

    handle = switzerland["DLH02A"]
    assert handle is not None
    assert (
        repr(handle.aircraft) == "Tail(icao24='3c6645', registration='D-AIRE',"
        " typecode='A321', flag='🇩🇪', category='>20t')"
    )

    handle = switzerland["4baa61"]
    assert handle is not None
    assert handle.callsign == "THY7WR"

    def min_altitude(flight: Flight) -> float:
        return flight.altitude_min  # type: ignore

    selected = max(switzerland, key=min_altitude)
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
    subset = switzerland[["EXS33W", "4009f9"]]
    assert subset is not None
    assert len(subset) == 4

    s_0 = switzerland[0]
    assert s_0 is not None
    assert s_0.callsign == "SAA260"

    subset = switzerland[:2]
    assert subset is not None
    assert subset.callsigns == {"SAA260", "SAA261"}
    assert subset.icao24 == {"00b0ed"}


def test_aircraft() -> None:
    subset = switzerland[["EXS33W", "4009f9"]]
    expected = {"A320", "B733"}
    assert subset is not None
    assert set(f.max("typecode") for f in subset.aircraft_data()) == expected


def high_altitude(flight: Flight) -> bool:
    return flight.altitude_min > 35000  # type: ignore


def test_chaining() -> None:
    sw_filtered = (
        switzerland.between("2018-08-01", "2018-08-02")  # type: ignore
        .inside_bbox(eurofirs["LSAS"])
        .assign_id()
        .pipe(high_altitude)
        .resample("10s")
        .filter()
        .filter(altitude=53)
        .unwrap()
        .airborne()
        .eval(max_workers=4)
    )

    assert len(sw_filtered) == 784
    assert sw_filtered.data.shape[0] > 80000
    assert min(len(f) for f in sw_filtered) == 60
    assert sw_filtered.data.altitude.max() == 47000.0

    # not smart to pop this out
    flight_id: str = sw_filtered.flight_ids.pop()
    handle = sw_filtered[flight_id]
    assert handle is not None
    assert handle.callsign == flight_id.split("_")[0]


def test_chaining_with_lambda() -> None:
    sw_filtered = (
        switzerland.between("2018-08-01", "2018-08-02")  # type: ignore
        .inside_bbox(eurofirs["LSAS"])
        .assign_id()
        .pipe(lambda flight: flight.altitude_min > 35000)
        .resample("10s")
        .filter()
        .filter(altitude=53)
        .unwrap()
        .airborne()
        .eval(max_workers=4)
    )

    assert len(sw_filtered) == 784
    assert sw_filtered.data.shape[0] > 80000
    assert min(len(f) for f in sw_filtered) == 60
    assert sw_filtered.data.altitude.max() == 47000.0


def test_none() -> None:
    assert switzerland.query("altitude > 60000") is None
    after = switzerland.after("2018-08-01")
    assert after is not None
    assert after.before("2018-08-01") is None
    assert switzerland.iterate_lazy().query("altitude > 60000").eval() is None


def test_aggregate() -> None:
    s_0 = switzerland[0]
    assert s_0 is not None
    s_0 = s_0.compute_xy()
    x_max = s_0.data.x.max()
    x_min = s_0.data.x.min()
    y_max = s_0.data.y.max()
    y_min = s_0.data.y.min()
    resolution = {"x": 2e2, "y": 5e3}
    expected_shape = (
        int(abs(x_max // resolution["x"]) + abs(x_min // resolution["x"]) + 1),
        int(abs(y_max // resolution["y"]) + abs(y_min // resolution["y"]) + 1),
    )
    output_shape = (
        s_0.agg_xy(resolution, icao24="nunique").to_xarray().icao24.values.shape
    )
    assert output_shape == expected_shape


def test_sub() -> None:
    s_0 = switzerland[0]
    assert s_0 is not None and s_0.icao24 is not None
    diff = switzerland - s_0.icao24
    assert diff is not None
    assert s_0.icao24 not in diff.icao24

    diff = switzerland - ["EXS33W", "4009f9"]
    assert diff is not None
    assert {"EXS33W", "4009f9"} not in diff.icao24

    f_BAW585E = switzerland["BAW585E"]
    assert f_BAW585E is not None
    f_BAW881V = switzerland["BAW881V"]
    assert f_BAW881V is not None
    f_BAW84ZV = switzerland["BAW84ZV"]
    assert f_BAW84ZV is not None

    diff = switzerland - f_BAW585E
    assert diff is not None
    assert "BAW585E" not in diff.callsigns

    assert diff["4009f9"] is not None
    assert diff["4009f9"].callsign == {"BAW881V", "BAW84ZV"}
    assert len(f_BAW881V) + len(f_BAW84ZV) == len(diff["4009f9"])

    sw_id = switzerland.assign_id().eval()
    assert sw_id is not None

    flight = sw_id["500142"]
    assert flight is not None
    assert isinstance(flight.flight_id, set)
    assert flight.callsign == "T7STK"

    flight_deleted = sw_id["T7STK_1158"]
    assert flight_deleted is not None

    diff = sw_id - flight_deleted
    assert diff is not None
    assert diff["T7STK"] is not None
    assert diff["T7STK"].start >= flight_deleted.stop

    assert (sw_id - sw_id) is None
    diff = sw_id - diff
    assert diff is not None
    assert diff.icao24 == {"500142"}
