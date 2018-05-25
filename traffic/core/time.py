from datetime import datetime, timedelta
from numbers import Number
from typing import Union

import maya

timelike = Union[str, Number, datetime]
time_or_delta = Union[timelike, timedelta]

def to_datetime(time: timelike) -> datetime:
    if isinstance(time, str):
        time = maya.parse(time)  # type: ignore
    if isinstance(time, maya.core.MayaDT):  # type: ignore
        time = time.epoch
    if isinstance(time, Number):
        time = datetime.fromtimestamp(time)  # type: ignore
    return time  # type: ignore

def round_time(time: timelike, how: str='before',
               date_delta: timedelta=timedelta(hours=1)) -> datetime:

    dt = to_datetime(time)

    round_to = date_delta.total_seconds()
    seconds = (dt - dt.min).seconds

    if how == 'after':
        rounding = (seconds + round_to) // round_to * round_to
    elif how == 'before':
        rounding = seconds // round_to * round_to
    else:
        raise ValueError("parameter how must be `before` or `after`")

    return dt + timedelta(0, rounding - seconds, -dt.microsecond)
