from datetime import datetime, timedelta
from numbers import Number
from typing import Iterator, Tuple, Union

import numpy as np

import maya
import pandas as pd

timelike = Union[str, Number, datetime, pd.Timestamp]
time_or_delta = Union[timelike, timedelta]
timetuple = Tuple[datetime, datetime, datetime, datetime]


def to_datetime(time: timelike) -> datetime:
    if isinstance(time, pd.Timestamp):
        time = time.to_pydatetime()
    if isinstance(time, str):
        time = maya.parse(time)  # type: ignore
    if isinstance(time, maya.core.MayaDT):  # type: ignore
        time = time.epoch
    if isinstance(time, Number):
        time = datetime.fromtimestamp(time)  # type: ignore
    return time  # type: ignore


def round_time(
    time: timelike, how: str = "before", by: timedelta = timedelta(hours=1)
) -> datetime:

    dt = to_datetime(time)

    round_to = by.total_seconds()
    seconds = (dt - dt.min).seconds

    if how == "after":
        rounding = (seconds + round_to) // round_to * round_to
    elif how == "before":
        rounding = seconds // round_to * round_to
    else:
        raise ValueError("parameter how must be `before` or `after`")

    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def split_times(
    before: datetime, after: datetime, by: timedelta = timedelta(hours=1)
) -> Iterator[timetuple]:

    before_hour = round_time(before, by=by)
    seq = np.arange(before_hour, after + by, by).astype(datetime)

    for bh, ah in zip(seq[:-1], seq[1:]):
        yield (before, after, bh, ah)
