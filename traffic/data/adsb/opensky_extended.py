"""
[hadoop-1:21000] > describe rollcall_replies_data4;
+----------------------+-------------------+---------+
| name                 | type              | comment |
+----------------------+-------------------+---------+
| sensors              | array<struct<     |         |
|                      |   serial:int,     |         |
|                      |   mintime:double, |         |
|                      |   maxtime:double  |         |
|                      | >>                |         |
| rawmsg               | string            |         |
| mintime              | double            |         |
| maxtime              | double            |         |
| msgcount             | bigint            |         |
| icao24               | string            |         |
| message              | string            |         |
| isid                 | boolean           |         |
| flightstatus         | tinyint           |         |
| downlinkrequest      | tinyint           |         |
| utilitymsg           | tinyint           |         |
| interrogatorid       | tinyint           |         |
| identifierdesignator | tinyint           |         |
| valuecode            | smallint          |         |
| altitude             | double            |         |
| identity             | string            |         |
| hour                 | int               |         |
+----------------------+-------------------+---------+
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional, Union

import pandas as pd

from ...core.time import split_times, timelike, to_datetime

_columns = [
    # "sensors", keep commented, array<*>
    "rawmsg",
    "mintime",
    "maxtime",
    "msgcount",
    "icao24",
    "message",
    "isid",
    "flightstatus",
    "downlinkrequest",
    "utilitymsg",
    "interrogatorid",
    "identifierdesignator",
    "valuecode",
    "altitude",
    "identity",
    "hour",
]

_request = (
    "select {columns} from rollcall_replies_data4 {other_tables} "
    "where hour>={before_hour} and hour<{after_hour} "
    "and rollcall_replies_data4.mintime>={before_time} "
    "and rollcall_replies_data4.maxtime<{after_time} "
    "{other_params}"
)


def extended(
    self,
    before: timelike,
    after: Optional[timelike] = None,
    *args,
    date_delta: timedelta = timedelta(hours=1),
    icao24: Optional[Union[str, Iterable[str]]] = None,
    serials: Optional[Iterable[int]] = None,
    other_tables: str = "",
    other_params: str = "",
    progressbar: Callable[[Iterable], Iterable] = iter,
    cached: bool = True,
):
    columns = (
        "rollcall_replies_data4.mintime, "
        "rollcall_replies_data4.maxtime, "
        "rawmsg, msgcount, icao24, message, identity, hour"
    )

    before = to_datetime(before)
    if after is not None:
        after = to_datetime(after)
    else:
        after = before + timedelta(days=1)

    if isinstance(icao24, str):
        other_params += "and icao24='{}' ".format(icao24)

    elif isinstance(icao24, Iterable):
        icao24 = ",".join("'{}'".format(c) for c in icao24)
        other_params += "and icao24 in ({}) ".format(icao24)

    if isinstance(serials, Iterable):
        other_tables += ", rollcall_replies_data4.sensors s "
        other_params += "and s.serial in {} ".format(tuple(serials))
        columns = "s.serial, s.mintime as time, " + columns
    else:
        raise NotImplementedError()

    other_params += "and message is not null "
    sequence = list(split_times(before, after, date_delta))
    cumul = []

    for bt, at, bh, ah in progressbar(sequence):

        logging.info(
            f"Sending request between time {bt} and {at} "
            f"and hour {bh} and {ah}"
        )

        request = _request.format(
            columns=columns,
            before_time=bt.timestamp(),
            after_time=at.timestamp(),
            before_hour=bh.timestamp(),
            after_hour=ah.timestamp(),
            other_tables=other_tables,
            other_params=other_params,
        )

        df = self._impala(request, cached)

        if df is None:
            continue

        if df.hour.dtype == object:
            df = df[df.hour != "hour"]

        for column_name in ["mintime", "maxtime", "time"]:
            df[column_name] = (
                df[column_name].astype(float).apply(datetime.fromtimestamp)
            )

        df.icao24 = df.icao24.apply(
            lambda x: "{:0>6}".format(hex(int(str(x), 16))[2:])
        )

        cumul.append(df)

    if len(cumul) == 0:
        return None

    return pd.concat(cumul)
