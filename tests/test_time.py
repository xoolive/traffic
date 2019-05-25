from datetime import datetime, timedelta, timezone

from traffic.core.time import round_time, split_times, to_datetime


def test_datetime() -> None:
    assert f"{to_datetime('2017-01-14')}" == "2017-01-14 00:00:00+00:00"
    assert f"{to_datetime('2017-01-14 12:00Z')}" == "2017-01-14 12:00:00+00:00"
    assert f"{to_datetime(1484395200)}" == "2017-01-14 12:00:00+00:00"
    assert (
        f"{to_datetime(datetime(2017, 1, 14, 12, tzinfo=timezone.utc))}"
        == "2017-01-14 12:00:00+00:00"
    )


def test_roundtime() -> None:
    assert f"{round_time('2018-01-12 13:30')}" == "2018-01-12 13:00:00+00:00"
    assert f"{round_time('2018-01-12 13:00')}" == "2018-01-12 13:00:00+00:00"
    assert (
        f"{round_time('2018-01-12 13:00', by=timedelta(hours=2))}"
        == "2018-01-12 12:00:00+00:00"
    )
    assert (
        f"{round_time('2018-01-12 13:00', how='after', by=timedelta(hours=2))}"
        == "2018-01-12 14:00:00+00:00"
    )


def test_splittime() -> None:
    assert next(
        f"{t1}, {t2}, {h1}, {h2}"
        for t1, t2, h1, h2 in split_times(
            to_datetime("2018-01-12 12:00"),
            to_datetime("2018-01-12 14:00"),
            by=timedelta(minutes=30),
        )
    ) == (
        "2018-01-12 12:00:00+00:00, 2018-01-12 14:00:00+00:00, "
        "2018-01-12 12:00:00+00:00, 2018-01-12 12:30:00+00:00"
    )
