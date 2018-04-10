from datetime import datetime, timedelta
from typing import Union

import maya

timelike = Union[str, int, datetime]
time_or_delta = Union[timelike, timedelta]

def to_datetime(time: timelike) -> datetime:
    if isinstance(time, str):
        time = maya.parse(time)
    if isinstance(time, maya.core.MayaDT):
        time = time.epoch
    if isinstance(time, int):
        time = datetime.fromtimestamp(time)
    return time

