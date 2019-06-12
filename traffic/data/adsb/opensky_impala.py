import hashlib
import logging
import re
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, Union

import pandas as pd
import paramiko
from shapely.geometry.base import BaseGeometry
from tqdm.autonotebook import tqdm

from ...core import Flight, Traffic
from ...core.time import split_times, timelike, to_datetime


class ImpalaError(Exception):
    pass


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

    stdin: paramiko.ChannelFile
    stdout: paramiko.ChannelFile
    stderr: paramiko.ChannelFile  # actually ChannelStderrFile

    def __init__(self, username: str, password: str, cache_dir: Path) -> None:

        self.username = username
        self.password = password
        self.connected = False
        self.cache_dir = cache_dir
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        if username == "" or password == "":
            self.auth = None
        else:
            self.auth = (username, password)

    def clear_cache(self) -> None:  # coverage: ignore
        """Clear cache files for OpenSky.

        The directory containing cache files tends to clog after a while.
        """
        for file in self.cache_dir.glob("*"):
            file.unlink()

    @staticmethod
    def _read_cache(cachename: Path) -> Optional[pd.DataFrame]:

        logging.info("Reading request in cache {}".format(cachename))
        with cachename.open("r") as fh:
            s = StringIO()
            count = 0
            for line in fh.readlines():
                # -- no pretty-print style cache (option -B)
                if re.search("\t", line):  # noqa: W605
                    # don't ask why re.match does not work
                    count += 1
                    s.write(re.sub(" *\t *", ",", line))  # noqa: W605
                    s.write("\n")
                # -- pretty-print style cache
                if re.match("\|.*\|", line):  # noqa: W605
                    count += 1
                    if "," in line:  # this may happen on 'describe table'
                        return_df = False
                        break
                    s.write(re.sub(" *\| *", ",", line)[1:-2])  # noqa: W605
                    s.write("\n")
            else:
                return_df = True

            if not return_df:
                fh.seek(0)
                return "".join(fh.readlines())

            if count > 0:
                s.seek(0)
                # otherwise pandas would parse 1234e5 as 123400000.0
                df = pd.read_csv(s, dtype={"icao24": str, "callsign": str})
                if df.shape[0] > 0:
                    return df

        with cachename.open("r") as fh:
            output = fh.readlines()
            if any(elt.startswith("ERROR:") for elt in output):
                msg = "".join(output[:-1])
                cachename.unlink()
                raise ImpalaError(msg)

        return None

    @staticmethod
    def _format_dataframe(
        df: pd.DataFrame, nautical_units=True
    ) -> pd.DataFrame:
        """
        This function converts types, strips spaces after callsigns and sorts
        the DataFrame by timestamp.

        For some reason, all data arriving from OpenSky are converted to
        units in metric system. Optionally, you may convert the units back to
        nautical miles, feet and feet/min.

        """

        if "callsign" in df.columns and df.callsign.dtype == object:
            df.callsign = df.callsign.str.strip()

        if nautical_units:
            df.altitude = df.altitude / 0.3048
            if "geoaltitude" in df.columns:
                df.geoaltitude = df.geoaltitude / 0.3048
            if "groundspeed" in df.columns:
                df.groundspeed = df.groundspeed / 1852 * 3600
            if "vertical_rate" in df.columns:
                df.vertical_rate = df.vertical_rate / 0.3048 * 60

        df.timestamp = pd.to_datetime(df.timestamp * 1e9).dt.tz_localize("utc")

        if "last_position" in df.columns:
            df = df.query("last_position == last_position").assign(
                last_position=pd.to_datetime(
                    df.last_position * 1e9
                ).dt.tz_localize("utc")
            )

        return df.sort_values("timestamp")

    def _connect(self) -> None:  # coverage: ignore
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
        self.stdin, self.stdout, self.stderr = client.exec_command(
            "-B", bufsize=-1, get_pty=True
        )
        self.connected = True
        total = ""
        while len(total) == 0 or total[-10:] != ":21000] > ":
            b = self.stdout.channel.recv(256)
            total += b.decode()

    def _impala(
        self, request: str, columns: str, cached: bool = True
    ) -> Optional[pd.DataFrame]:  # coverage: ignore

        digest = hashlib.md5(request.encode("utf8")).hexdigest()
        cachename = self.cache_dir / digest

        if cachename.exists() and not cached:
            cachename.unlink()

        if not cachename.exists():
            if not self.connected:
                self._connect()

            logging.info("Sending request: {}".format(request))
            # bug fix for when we write a request with """ starting with \n
            request = request.replace("\n", " ")
            self.stdin.channel.send(request + ";\n")
            total = ""
            while len(total) == 0 or total[-10:] != ":21000] > ":
                b = self.stdout.channel.recv(256)
                total += b.decode()
            with cachename.open("w") as fh:
                if columns is not None:
                    fh.write(re.sub(", ", "\t", columns))
                    fh.write("\n")
                fh.write(total)

        return self._read_cache(cachename)

    @staticmethod
    def _format_history(df: pd.DataFrame) -> pd.DataFrame:
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

        df.icao24 = (
            df.icao24.apply(int, base=16)
            .apply(hex)
            .str.slice(2)
            .str.pad(6, fillchar="0")
        )

        if df.onground.dtype != bool:
            df.onground = df.onground == "true"
            df.alert = df.alert == "true"
            df.spi = df.spi == "true"

        # better (to me) formalism about columns
        return df.rename(
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

    def history(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        *args,  # more reasonable to be explicit about arguments
        date_delta: timedelta = timedelta(hours=1),
        callsign: Union[None, str, Iterable[str]] = None,
        icao24: Union[None, str, Iterable[str]] = None,
        serials: Union[None, str, Iterable[str]] = None,
        bounds: Union[
            BaseGeometry, Tuple[float, float, float, float], None
        ] = None,
        cached: bool = True,
        count: bool = False,
        other_tables: str = "",
        other_params: str = "",
        progressbar: Callable[[Iterable], Iterable] = iter,
    ) -> Optional[Union[Traffic, Flight]]:

        """Get Traffic from the OpenSky Impala shell.

        The method builds appropriate SQL requests, caches results and formats
        data into a proper pandas DataFrame. Requests are split by hour (by
        default) in case the connection fails.

        Args:
            start: a string, epoch or datetime
            stop (optional): a string, epoch or datetime, by default, one day
            after start
            date_delta (optional): how to split the requests (default: one day)
            callsign (optional): a string or a list of strings (default: empty)
            icao24 (optional): a string or a list of strings identifying the
            transponder code of the aircraft (default: empty)
            serials (optional): a string or a list of strings identifying the
            sensors receiving the data. (default: empty)
            bounds (optional): a shape (requires the bounds attribute) or a
            tuple of floats (west, south, east, north) to put a geographical
            limit on the request. (default: empty)
            cached (boolean): whether to look first whether the request has been
            cached (default: True)
            count (boolean): add a column stating how many sensors received each
            line (default: False)

        Returns:
            a Traffic structure wrapping the dataframe

        """

        return_flight = False
        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        if progressbar == iter and stop - start > timedelta(hours=1):
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
        sequence = list(split_times(start, stop, date_delta))
        columns = ", ".join(self._impala_columns)
        parse_columns = ", ".join(self._impala_columns)

        if count is True:
            other_params += "group by " + columns
            columns = "count(*) as count, " + columns
            parse_columns = "count, " + parse_columns
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

            df = self._impala(request, columns=parse_columns, cached=cached)

            if df is None:
                continue

            df = self._format_history(df)

            if "last_position" in df.columns:
                if df.query("last_position == last_position").shape[0] == 0:
                    continue

            df = self._format_dataframe(df)

            cumul.append(df)

        if len(cumul) == 0:
            return None

        df = pd.concat(cumul, sort=True).sort_values("timestamp")

        if count is True:
            df = df.assign(count=lambda df: df["count"].astype(int))

        if return_flight:
            return Flight(df)

        return Traffic(df)

    def extended(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        *args,  # more reasonable to be explicit about arguments
        date_delta: timedelta = timedelta(hours=1),
        icao24: Union[None, str, Iterable[str]] = None,
        serials: Union[None, int, Iterable[int]] = None,
        other_tables: str = "",
        other_params: str = "",
        progressbar: Callable[[Iterable], Iterable] = iter,
        cached: bool = True,
    ) -> pd.DataFrame:
        """Get EHS message from the OpenSky Impala shell.

        The method builds appropriate SQL requests, caches results and formats
        data into a proper pandas DataFrame. Requests are split by hour (by
        default) in case the connection fails.

        Args:
            start: a string, epoch or datetime
            stop (optional): a string, epoch or datetime, by default, one day
            after start
            date_delta (optional): how to split the requests (default: one day)
            icao24 (optional): a string or a list of strings identifying the
            transponder code of the aircraft (default: empty)
            serials (optional): an int or a list of int identifying the
            sensors receiving the data. (default: empty)
            cached (boolean): whether to look first whether the request has been
            cached (default: True)

        Returns:
            a Traffic structure wrapping the dataframe

        """

        _request = (
            "select {columns} from rollcall_replies_data4 r {other_tables} "
            "where hour>={before_hour} and hour<{after_hour} "
            "and r.mintime>={before_time} and r.mintime<{after_time} "
            "{other_params}"
        )

        columns = (
            "r.mintime, r.maxtime, "
            "rawmsg, msgcount, icao24, message, altitude, identity, hour"
        )

        parse_columns = (
            "mintime, maxtime, "
            "rawmsg, msgcount, icao24, message, altitude, identity, hour"
        )

        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        if isinstance(icao24, str):
            other_params += "and icao24='{}' ".format(icao24)
        elif isinstance(icao24, Iterable):
            icao24 = ",".join("'{}'".format(c) for c in icao24)
            other_params += "and icao24 in ({}) ".format(icao24)

        if isinstance(serials, Iterable):
            other_tables += ", rollcall_replies_data4.sensors s "
            other_params += "and s.serial in {} ".format(tuple(serials))
            columns = "s.serial, s.mintime as time, " + columns
            parse_columns = "serial, time, " + parse_columns
        elif isinstance(serials, int):
            other_tables += ", rollcall_replies_data4.sensors s "
            other_params += "and s.serial = {} ".format((serials))
            columns = "s.serial, s.mintime as time, " + columns
            parse_columns = "serial, time, " + parse_columns

        other_params += "and message is not null "
        sequence = list(split_times(start, stop, date_delta))
        cumul = []

        for bt, at, bh, ah in progressbar(sequence):

            logging.info(
                f"Sending request between time {bt} and {at} "
                f"and hour {bh} and {ah}"
            )

            request = _request.format(
                columns=columns,
                before_time=int(bt.timestamp()),
                after_time=int(at.timestamp()),
                before_hour=bh.timestamp(),
                after_hour=ah.timestamp(),
                other_tables=other_tables,
                other_params=other_params,
            )

            df = self._impala(request, columns=parse_columns, cached=cached)

            if df is None:
                continue

            if df.hour.dtype == object:
                df = df[df.hour != "hour"]

            for column_name in ["mintime", "maxtime"]:
                df[column_name] = pd.to_datetime(
                    df[column_name].astype(float) * 1e9
                ).dt.tz_localize("utc")

            df.icao24 = (
                df.icao24.apply(int, base=16)
                .apply(hex)
                .str.slice(2)
                .str.pad(6, fillchar="0")
            )
            df.altitude = df.altitude.astype(float) / 0.3048

            cumul.append(df)

        if len(cumul) == 0:
            return None

        return pd.concat(cumul).sort_values("mintime")


# below this line is only helpful references
# ------------------------------------------

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
