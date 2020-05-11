import warnings
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Iterator, Tuple, Union

import numpy as np
import pandas as pd

timelike = Union[str, Real, datetime, pd.Timestamp]
deltalike = Union[None, str, Real, timedelta, pd.Timedelta]

time_or_delta = Union[timelike, timedelta]
timetuple = Tuple[datetime, datetime, datetime, datetime]


def to_timedelta(delta: deltalike, **kwargs) -> Union[timedelta, pd.Timedelta]:
    if isinstance(delta, Real):
        delta = timedelta(seconds=float(delta))
    if isinstance(delta, str):
        delta = pd.Timedelta(delta)
    if delta is None:
        delta = timedelta(**kwargs)
    return delta


def to_datetime(time: timelike) -> datetime:
    if isinstance(time, str):
        time = pd.Timestamp(time, tz="utc")
    if isinstance(time, pd.Timestamp):
        time = time.to_pydatetime()
    if isinstance(time, Real):
        time = datetime.fromtimestamp(float(time), timezone.utc)
    if time.tzinfo is None:  # coverage: ignore
        warnings.warn(
            "This timestamp is tz-naive. Things may not work as expected. "
            "If you construct your timestamps manually, consider passing a "
            "string, which defaults to UTC. If you construct your timestamps "
            "automatically, look at the tzinfo (resp. tz) argument of the "
            "datetime (resp. pd.Timestamp) constructor."
        )
    return time


def round_time(
    time: timelike,
    how: str = "before",
    by: timedelta = timedelta(hours=1),  # noqa: B008
) -> datetime:

    dt = to_datetime(time)

    round_to = by.total_seconds()
    if dt.tzinfo is None:  # coverage: ignore
        seconds = (dt - dt.min).seconds
    else:
        seconds = (dt - dt.min.replace(tzinfo=timezone.utc)).seconds

    if how == "after":
        rounding = (seconds + round_to) // round_to * round_to
    elif how == "before":
        rounding = seconds // round_to * round_to
    else:  # coverage: ignore
        raise ValueError("parameter how must be `before` or `after`")

    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def split_times(
    before: datetime,
    after: datetime,
    by: timedelta = timedelta(hours=1),  # noqa: B008
) -> Iterator[timetuple]:

    before_hour = round_time(before, by=by)
    seq = np.arange(before_hour, after + by, by, dtype=datetime)

    for bh, ah in zip(seq[:-1], seq[1:]):
        yield (before, after, bh, ah)
