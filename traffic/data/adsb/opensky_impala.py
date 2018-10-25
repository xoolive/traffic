import hashlib
import logging
import re
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union, cast

import pandas as pd
import paramiko
from tqdm.autonotebook import tqdm

from ...core import Flight, Traffic
from ...core.time import round_time, split_times, timelike, to_datetime
from ..basic.airport import Airport


class Impala(object):

    _impala_columns = [
        "time",
        "icao24",
        "lat",
        "lon",
        "velocity",
        "heading",
        "vertrate",
        "callsign",
        "onground",
        "alert",
        "spi",
        "squawk",
        "baroaltitude",
        "geoaltitude",
        "lastposupdate",
        "lastcontact",
        # "serials", keep commented, array<int>
        "hour",
    ]

    basic_request = (
        "select {columns} from state_vectors_data4 {other_tables} "
        "where hour>={before_hour} and hour<{after_hour} "
        "and time>={before_time} and time<{after_time} "
        "{other_params}"
    )

    def __init__(self, username: str, password: str, cache_dir: Path) -> None:

        self.username = username
        self.password = password
        self.connected = False
        self.cache_dir = cache_dir
        self.shell = None
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        if username == "" or password == "":
            self.auth = None
        else:
            self.auth = (username, password)

    def clear_cache(self) -> None:
        for file in self.cache_dir.glob("*"):
            file.unlink()

    @staticmethod
    def _format_dataframe(
        df: pd.DataFrame, nautical_units=True
    ) -> pd.DataFrame:

        df.callsign = df.callsign.str.strip()

        if nautical_units:
            df.altitude = df.altitude / 0.3048
            if 'geoaltitude' in df.columns:
                df.geoaltitude = df.geoaltitude / 0.3048
            if 'groundspeed' in df.columns:
                df.groundspeed = df.groundspeed / 1852 * 3600
            if 'vertical_rate' in df.columns:
                df.vertical_rate = df.vertical_rate / 0.3048 * 60

        df.timestamp = df.timestamp.apply(datetime.fromtimestamp)

        # warning is raised here: SettingWithCopyWarning:
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_indexer,col_indexer] = value instead
        pd.options.mode.chained_assignment = None
        if 'last_position' in df.columns:
            df = df.loc[df.last_position.notnull()]  # do we really miss much?
            df.last_position = df.last_position.apply(datetime.fromtimestamp)

        return df.sort_values("timestamp")

    def _connect(self) -> None:
        if self.username == "" or self.password == "":
            raise RuntimeError("This method requires authentication.")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            "data.opensky-network.org",
            port=2230,
            username=self.username,
            password=self.password,
            look_for_keys=False,
            allow_agent=False,
            compress=True,
        )
        self.shell = client.invoke_shell()
        self.connected = True
        total = ""
        while len(total) == 0 or total[-19:] != "[hadoop-1:21000] > ":
            b = self.shell.recv(256)
            total += b.decode()

    def _impala(
        self, request: str, cached: bool = True
    ) -> Optional[pd.DataFrame]:

        digest = hashlib.md5(request.encode("utf8")).hexdigest()
        cachename = self.cache_dir / digest

        if not cached:
            cachename.unlink()

        if not cachename.exists():
            if not self.connected:
                self._connect()

            # Now we are connected so self.c is not None
            self.shell = cast(Any, self.shell)

            logging.info("Sending request: {}".format(request))
            self.shell.send(request + ";\n")
            total = ""
            while len(total) == 0 or total[-19:] != "[hadoop-1:21000] > ":
                b = self.shell.recv(256)
                total += b.decode()
            with cachename.open("w") as fh:
                fh.write(total)

        logging.info("Reading request in cache {}".format(cachename))
        with cachename.open("r") as fh:
            s = StringIO()
            count = 0
            for line in fh.readlines():
                if re.match("\|.*\|", line):
                    count += 1
                    s.write(re.sub(" *\| *", ",", line)[1:-2])
                    s.write("\n")
            if count > 0:
                s.seek(0)
                df = pd.read_csv(s)
                return df

        return None

    def history(
        self,
        before: timelike,
        after: Optional[timelike] = None,
        *args,
        date_delta: timedelta = timedelta(hours=1),
        callsign: Optional[Union[str, Iterable[str]]] = None,
        icao24: Optional[Union[str, Iterable[str]]] = None,
        serials: Optional[Iterable[int]] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        other_tables: str = "",
        other_params: str = "",
        progressbar: Callable[[Iterable], Iterable] = iter,
        cached: bool = True,
        count: bool = False,
    ) -> Optional[Union[Traffic, Flight]]:

        return_flight = False
        before = to_datetime(before)
        if after is not None:
            after = to_datetime(after)
        else:
            after = before + timedelta(days=1)

        if progressbar == iter and after - before > timedelta(hours=1):
            progressbar = tqdm

        if isinstance(serials, Iterable):
            other_tables += ", state_vectors_data4.serials s "
            other_params += "and s.ITEM in {} ".format(tuple(serials))

        if isinstance(icao24, str):
            other_params += "and icao24='{}' ".format(icao24)

        elif isinstance(icao24, Iterable):
            icao24 = ",".join("'{}'".format(c) for c in icao24)
            other_params += "and icao24 in ({}) ".format(icao24)

        if isinstance(callsign, str):
            other_params += "and callsign='{:<8s}' ".format(callsign)
            return_flight = True

        elif isinstance(callsign, Iterable):
            callsign = ",".join("'{:<8s}'".format(c) for c in callsign)
            other_params += "and callsign in ({}) ".format(callsign)

        if bounds is not None:
            try:
                # thinking of shapely bounds attribute (in this order)
                # I just don't want to add the shapely dependency here
                west, south, east, north = bounds.bounds  # type: ignore
            except AttributeError:
                west, south, east, north = bounds

            other_params += "and lon>={} and lon<={} ".format(west, east)
            other_params += "and lat>={} and lat<={} ".format(south, north)

        cumul = []
        sequence = list(split_times(before, after, date_delta))
        columns = ", ".join(self._impala_columns)

        if count is True:
            other_params += "group by " + columns
            columns = "count(*) as count, " + columns
            other_tables += ", state_vectors_data4.serials s"

        for bt, at, bh, ah in progressbar(sequence):

            logging.info(
                f"Sending request between time {bt} and {at} "
                f"and hour {bh} and {ah}"
            )

            request = self.basic_request.format(
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

            df = df.drop(["lastcontact"], axis=1)

            if df.lat.dtype == object:
                df = df[df.lat != "lat"]  # header is regularly repeated
            # restore all types
            for column_name in [
                "lat",
                "lon",
                "velocity",
                "heading",
                "vertrate",
                "baroaltitude",
                "geoaltitude",
                "lastposupdate",
                # "lastcontact",
            ]:
                df[column_name] = df[column_name].astype(float)

            for column_name in ["time", "hour"]:
                df[column_name] = df[column_name].astype(int)

            df.icao24 = df.icao24.apply(
                lambda x: "{:0>6}".format(hex(int(str(x), 16))[2:])
            )

            if df.onground.dtype != bool:
                df.onground = df.onground == "true"
                df.alert = df.alert == "true"
                df.spi = df.spi == "true"

            # better (to me) formalism about columns
            df = df.rename(
                columns={
                    "lat": "latitude",
                    "lon": "longitude",
                    "heading": "track",
                    "velocity": "groundspeed",
                    "vertrate": "vertical_rate",
                    "baroaltitude": "altitude",
                    "time": "timestamp",
                    "lastposupdate": "last_position",
                }
            )

            df = self._format_dataframe(df)
            cumul.append(df)

        if len(cumul) == 0:
            return None

        if return_flight:
            return Flight(pd.concat(cumul, sort=True))

        return Traffic(pd.concat(cumul, sort=True))

    def sensors(
        self,
        before: timelike,
        after: timelike,
        bounds: Tuple[float, float, float, float],
    ) -> Optional[pd.DataFrame]:

        before = to_datetime(before)
        after = to_datetime(after)

        before_hour = round_time(before, "before")
        after_hour = round_time(after, "after")

        try:
            # thinking of shapely bounds attribute (in this order)
            # I just don't want to add the shapely dependency here
            west, south, east, north = bounds.bounds  # type: ignore
        except AttributeError:
            west, south, east, north = bounds

        other_params = "and lon>={} and lon<={} ".format(west, east)
        other_params += "and lat>={} and lat<={} ".format(south, north)

        query = self.basic_request.format(
            columns="icao24, callsign, s.ITEM as serial, count(*) as count",
            other_tables=", state_vectors_data4.serials s",
            before_time=before.timestamp(),
            after_time=after.timestamp(),
            before_hour=before_hour.timestamp(),
            after_hour=after_hour.timestamp(),
            other_params=other_params + "group by icao24, callsign, s.ITEM",
        )

        logging.info(f"Sending request: {query}")
        df = self._impala(query)
        if df is None:
            return None

        df = df[df["count"] != "count"]
        df["count"] = df["count"].astype(int)

        return df

    def at_airport(
        self,
        before: timelike,
        after: timelike,
        airport: Union[Airport, str],
        count: bool = False,
    ) -> Optional[pd.DataFrame]:

        before = to_datetime(before)
        after = to_datetime(after)

        before_hour = round_time(before, how="before")
        after_hour = round_time(after, how="after")

        if isinstance(airport, str):
            from traffic.data import airports

            airport = cast(Airport, airports[airport])

        other_params = (
            "and lat<={airport_latmax} and lat>={airport_latmin} "
            "and lon<={airport_lonmax} and lon>={airport_lonmin} "
            "and baroaltitude<=1000 "
            "group by icao24, callsign"
        ).format(
            airport_latmax=airport.lat + 0.1,
            airport_latmin=airport.lat - 0.1,
            airport_lonmax=airport.lon + 0.1,
            airport_lonmin=airport.lon - 0.1,
        )

        columns = "icao24, callsign"
        other_tables = ""
        if count is True:
            columns = "count(*) as count, s.ITEM as serial, " + columns
            other_tables += ", state_vectors_data4.serials s"
            other_params += ", s.ITEM"

        request = self.basic_request.format(
            columns=columns,
            before_time=before.timestamp(),
            after_time=after.timestamp(),
            before_hour=before_hour.timestamp(),
            after_hour=after_hour.timestamp(),
            other_tables=other_tables,
            other_params=other_params,
        )

        df = self._impala(request)

        return df

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

        _request = (
            "select {columns} from rollcall_replies_data4 {other_tables} "
            "where hour>={before_hour} and hour<{after_hour} "
            "and rollcall_replies_data4.mintime>={before_time} "
            "and rollcall_replies_data4.maxtime<{after_time} "
            "{other_params}"
        )

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
