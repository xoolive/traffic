import pytest
from cartes.crs import Lambert93  # type: ignore

import pandas as pd
from traffic.algorithms.filters.aggressive import FilterClustering
from traffic.algorithms.filters.ekf import EKF
from traffic.algorithms.ground.kalman_taxiway import KalmanTaxiway
from traffic.data.samples import full_flight_short

full_flight_short = full_flight_short.assign(callsign="AFR34ZG")


def test_snappy() -> None:
    # Taking here some raw data which contains both groundspeed and track angle
    # values. They are also available on OpenSky trino database but not in the
    # regular state vectors, only in the position4 table.

    g1 = full_flight_short.first("20 min").query('bds == "06"')
    assert g1 is not None
    snappy_cdg = KalmanTaxiway(airport="LFPG", projection=Lambert93())
    h1 = g1.filter(snappy_cdg)
    assert h1 is not None

    g2 = full_flight_short.last("20 min").query('bds == "06"')
    assert g2 is not None
    snappy_tls = KalmanTaxiway(airport="LFBO", projection=Lambert93())
    h2 = g2.filter(snappy_tls)
    assert h2 is not None

    assert 0 < g1.distance(h1).lateral.max() < 0.1
    assert 0 < g2.distance(h2).lateral.max() < 0.1


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="TODO fix bug on Windows",
)
def test_ekf() -> None:
    subset = full_flight_short.compute_xy(projection=Lambert93()).query(
        'bds in ["05", "09"]'
    )
    assert subset is not None
    # TODO the filter should work with partial measurements
    f = subset.resample("1s")
    assert f is not None
    g = f.filter(EKF())
    distance = g.distance(f)

    # assert distance.lateral.max() <  # currently 0
    assert distance.vertical.max() < 50


def test_filter_clustering_onground_pyarrow() -> None:
    """FilterClustering must handle pyarrow-backed bool onground with NA.

    When pandas (2.x or 3.x) reads ADS-B data from parquet, boolean
    columns like ``onground`` use the ``bool[pyarrow]`` dtype.  After
    resampling, NA values appear.  ``FilterClustering`` calls
    ``diff().astype(bool)`` which raises
    ``TypeError: boolean value of NA is ambiguous`` unless NA values
    are handled before the cast.
    """
    n = 50
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01", periods=n, freq="1s", tz="UTC"
            ),
            "latitude": [48.0 + i * 0.001 for i in range(n)],
            "longitude": [2.0 + i * 0.001 for i in range(n)],
            "altitude": [35000.0] * n,
            "groundspeed": [450.0] * n,
            "track": [180.0] * n,
            "vertical_rate": [0.0] * n,
            # pyarrow bool with NA — reproduces post-resample state
            "onground": pd.array(
                [False] * 20 + [None] * 10 + [False] * 20,
                dtype="bool[pyarrow]",
            ),
        }
    )

    filt = FilterClustering()
    result = filt.apply(df)

    assert len(result) == n
    assert "onground" in result.columns
