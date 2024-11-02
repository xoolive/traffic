from cartes.crs import Lambert93  # type: ignore

from traffic.algorithms.filters.ekf import EKF
from traffic.algorithms.filters.ground import KalmanTaxiway
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


def test_ekf() -> None:
    f = (
        full_flight_short.compute_xy(projection=Lambert93())
        .query('bds in ["05", "09"]')  # type: ignore
        .resample("1s")  # TODO the filter should work with partial measurements
    )
    assert f is not None
    g = f.filter(EKF())
    distance = g.distance(f)

    # assert distance.lateral.max() <  # currently 0
    assert distance.vertical.max() < 50
