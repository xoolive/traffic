# fmt: off

import hashlib
import logging
import re
import string
import time
from datetime import timedelta
from io import StringIO
from pathlib import Path
from tempfile import gettempdir
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import pandas as pd
import paramiko
from pandas.errors import ParserError
from shapely.geometry.base import BaseGeometry
from tqdm.autonotebook import tqdm

from ...core import Flight, Traffic
from ...core.time import round_time, split_times, timelike, to_datetime
from .raw_data import RawData

# fmt: on


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

    _raw_tables = [
        "acas_data4",
        "allcall_replies_data4",
        "identification_data4",
        "operational_status_data4",
        "position_data4",
        "rollcall_replies_data4",
        "velocity_data4",
    ]

    basic_request = (
        "select {columns} from state_vectors_data4 {other_tables} "
        "{where_clause} hour>={before_hour} and hour<{after_hour} "
        "and time>={before_time} and time<{after_time} "
        "{other_params}"
    )

    _parseErrorMsg = """
    Error at parsing the cache file, moved to a temporary directory: {path}.
    Running the request again may help.

    Consider filing an issue detailing the request if the problem persists:
    https://github.com/xoolive/traffic/issues

    For more information, find below the error and the buggy line:
    """

    stdin: paramiko.ChannelFile
    stdout: paramiko.ChannelFile
    stderr: paramiko.ChannelFile  # actually ChannelStderrFile

    def __init__(
        self, username: str, password: str, cache_dir: Path, proxy_command: str
    ) -> None:

        self.username = username
        self.password = password
        self.proxy_command = proxy_command
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
                    count += 1
                    s.write(re.sub(" *\t *", ",", line))  # noqa: W605
                    s.write("\n")
                # -- pretty-print style cache
                if re.match(r"\|.*\|", line):  # noqa: W605
                    count += 1
                    if "," in line:  # this may happen on 'describe table'
                        return_df = False
                        break
                    s.write(re.sub(r" *\| *", ",", line)[1:-2])  # noqa: W605
                    s.write("\n")
            else:
                return_df = True

            if not return_df:
                fh.seek(0)
                return "".join(fh.readlines())

            if count > 0:
                s.seek(0)
                try:
                    # otherwise pandas would parse 1234e5 as 123400000.0
                    df = pd.read_csv(s, dtype={"icao24": str, "callsign": str})
                except ParserError as error:
                    for x in re.finditer(r"line (\d)+,", error.args[0]):
                        line_nb = int(x.group(1))
                        with cachename.open("r") as fh:
                            content = fh.readlines()[line_nb - 1]

                    new_path = Path(gettempdir()) / cachename.name
                    cachename.rename(new_path)
                    raise ImpalaError(
                        Impala._parseErrorMsg.format(path=new_path)
                        + (error + "\n" + content)
                    )

                if df.shape[0] > 0:
                    return df.drop_duplicates()

        error_msg: Optional[str] = None
        with cachename.open("r") as fh:
            output = fh.readlines()
            if any(elt.startswith("ERROR:") for elt in output):
                error_msg = "".join(output[:-1])

        if error_msg is not None:
            cachename.unlink()
            raise ImpalaError(error_msg)

        return None

    @staticmethod
    def _format_dataframe(df: pd.DataFrame,) -> pd.DataFrame:
        """
        This function converts types, strips spaces after callsigns and sorts
        the DataFrame by timestamp.

        For some reason, all data arriving from OpenSky are converted to
        units in metric system. Optionally, you may convert the units back to
        nautical miles, feet and feet/min.

        """

        if "callsign" in df.columns and df.callsign.dtype == object:
            df.callsign = df.callsign.str.strip()

        df.icao24 = (
            df.icao24.apply(int, base=16)
            .apply(hex)
            .str.slice(2)
            .str.pad(6, fillchar="0")
        )

        if "rawmsg" in df.columns and df.rawmsg.dtype != str:
            df.rawmsg = df.rawmsg.astype(str).str.strip()

        if "squawk" in df.columns:
            df.squawk = (
                df.squawk.astype(str)
                .str.split(".")
                .str[0]
                .replace({"nan": None})
            )

        time_dict: Dict[str, pd.Series] = dict()
        for colname in [
            "lastposupdate",
            "lastposition",
            "firstseen",
            "lastseen",
            "mintime",
            "maxtime",
            "time",
            "timestamp",
            "day",
            "hour",
        ]:
            if colname in df.columns:
                time_dict[colname] = pd.to_datetime(
                    df[colname] * 1e9
                ).dt.tz_localize("utc")

        return df.assign(**time_dict)

    def _connect(self) -> None:  # coverage: ignore
        if self.username == "" or self.password == "":
            raise RuntimeError("This method requires authentication.")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        extra_args = dict()

        if self.proxy_command != "":
            # for instance:
            #    "ssh -W data.opensky-network.org:2230 proxy_machine"
            # or "connect.exe -H proxy_ip:proxy_port %h %p"
            logging.info(f"Using ProxyCommand: {self.proxy_command}")
            extra_args["sock"] = paramiko.ProxyCommand(self.proxy_command)

        client.connect(
            "data.opensky-network.org",
            port=2230,
            username=self.username,
            password=self.password,
            look_for_keys=False,
            allow_agent=False,
            compress=True,
            **extra_args,
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
            # avoid messing lines in the cache file
            time.sleep(0.1)
            total = ""
            logging.info("Will be writing into {}".format(cachename))
            while len(total) == 0 or total[-10:] != ":21000] > ":
                b = self.stdout.channel.recv(256)
                total += b.decode()
            # There is no direct streaming into the cache file.
            # The reason for that is the connection may stall, your computer
            # may crash or the programme may exit abruptly in spite of your
            # (and my) best efforts to handle exceptions.
            # If data is streamed directly into the cache file, it is hard to
            # detect that it is corrupted and should be removed/overwritten.
            logging.info("Opening {}".format(cachename))
            with cachename.open("w") as fh:
                if columns is not None:
                    fh.write(re.sub(", ", "\t", columns))
                    fh.write("\n")
                fh.write(total)
            logging.info("Closing {}".format(cachename))

        return self._read_cache(cachename)

    @staticmethod
    def _format_history(
        df: pd.DataFrame, nautical_units: bool = True
    ) -> pd.DataFrame:

        if "lastcontact" in df.columns:
            df = df.drop(["lastcontact"], axis=1)

        if "lat" in df.columns and df.lat.dtype == object:
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
            # "lastposupdate",
            # "lastcontact",
        ]:
            if column_name in df.columns:
                df[column_name] = df[column_name].astype(float)

        if "onground" in df.columns and df.onground.dtype != bool:
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

        if nautical_units:
            df.altitude = (df.altitude / 0.3048).round(0)
            if "geoaltitude" in df.columns:
                df.geoaltitude = (df.geoaltitude / 0.3048).round(0)
            if "groundspeed" in df.columns:
                df.groundspeed = (df.groundspeed / 1852 * 3600).round(0)
            if "vertical_rate" in df.columns:
                df.vertical_rate = (df.vertical_rate / 0.3048 * 60).round(0)

        return df

    def request(
        self,
        request_pattern: str,
        start: timelike,
        stop: timelike,
        *args,  # more reasonable to be explicit about arguments
        columns: List[str],
        date_delta: timedelta = timedelta(hours=1),  # noqa: B008
        cached: bool = True,
        progressbar: Union[bool, Callable[[Iterable], Iterable]] = True,
    ) -> pd.DataFrame:
        """Splits and sends a custom request.

        Args:
            - **request_pattern**: a string containing the basic request you
              wish to make on Impala shell. Use {before_hour} and {after_hour}
              place holders to write your hour constraints: they will be
              automatically replaced by appropriate values.
            - **start**: a string (default to UTC), epoch or datetime (native
              Python or pandas)
            - **stop** (optional): a string (default to UTC), epoch or datetime
              (native Python or pandas), *by default, one day after start*
            - **columns**: the list of expected columns in the result. This
              helps naming the columns in the resulting dataframe.

        **Useful options**

            - **date_delta** (optional): a timedelta representing how to split
              the requests, *by default: per hour*
            - **cached** (boolean, default: True): switch to False to force a
              new request to the database regardless of the cached files;
              delete previous cache files;

        """

        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        if progressbar is True:
            if stop - start > date_delta:
                progressbar = tqdm
            else:
                progressbar = iter

        if progressbar is False:
            progressbar = iter

        progressbar = cast(Callable[[Iterable], Iterable], progressbar)

        cumul: List[pd.DataFrame] = []
        sequence = list(split_times(start, stop, date_delta))

        for bt, at, bh, ah in progressbar(sequence):
            logging.info(
                f"Sending request between time {bt} and {at} "
                f"and hour {bh} and {ah}"
            )

            request = request_pattern.format(
                before_time=bt.timestamp(),
                after_time=at.timestamp(),
                before_hour=bh.timestamp(),
                after_hour=ah.timestamp(),
            )

            df = self._impala(
                request, columns="\t".join(columns), cached=cached
            )

            if df is None:
                continue

            cumul.append(df)

        if len(cumul) == 0:
            return None

        return pd.concat(cumul, sort=True)

    def flightlist(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        *args,  # more reasonable to be explicit about arguments
        departure_airport: Union[None, str, Iterable[str]] = None,
        arrival_airport: Union[None, str, Iterable[str]] = None,
        airport: Union[None, str, Iterable[str]] = None,
        callsign: Union[None, str, Iterable[str]] = None,
        icao24: Union[None, str, Iterable[str]] = None,
        cached: bool = True,
        limit: Optional[int] = None,
        progressbar: Union[bool, Callable[[Iterable], Iterable]] = True,
    ) -> pd.DataFrame:
        """Lists flights departing or arriving at a given airport.

        You may pass requests based on time ranges, callsigns, aircraft, areas,
        serial numbers for receivers, or airports of departure or arrival.

        The method builds appropriate SQL requests, caches results and formats
        data into a proper pandas DataFrame. Requests are split by hour (by
        default) in case the connection fails.

        Args:
            - **start**: a string (default to UTC), epoch or datetime (native
              Python or pandas)
            - **stop** (optional): a string (default to UTC), epoch or datetime
              (native Python or pandas), *by default, one day after start*

            More arguments to filter resulting data:

            - **departure_airport** (optional): a string for the ICAO
              identifier of the airport. Selects flights departing from the
              airport between the two timestamps;
            - **arrival_airport** (optional): a string for the ICAO identifier
              of the airport. Selects flights arriving at the airport between
              the two timestamps;
            - **airport** (optional): a string for the ICAO identifier of the
              airport. Selects flights departing from or arriving at the
              airport between the two timestamps;
            - **callsign** (optional): a string or a list of strings (wildcards
              accepted, _ for any character, % for any sequence of characters);
            - **icao24** (optional): a string or a list of strings identifying i
              the transponder code of the aircraft;

            .. warning::

                - If both departure_airport and arrival_airport are set,
                  requested timestamps match the arrival time;
                - If airport is set, departure_airport and
                  arrival_airport cannot be specified (a RuntimeException is
                  raised).

            **Useful options for debug**

            - **cached** (boolean, default: True): switch to False to force a
              new request to the database regardless of the cached files;
              delete previous cache files;
            - **limit** (optional, int): maximum number of records requested
              LIMIT keyword in SQL.

        """

        query_str = (
            "select {columns} from flights_data4 "
            "where day >= {before_day} and day < {after_day} "
            "{other_params}"
        )
        columns = ", ".join(
            [
                "icao24",
                "firstseen",
                "estdepartureairport",
                "lastseen",
                "estarrivalairport",
                "callsign",
                "day",
            ]
        )

        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        if progressbar is True:
            if stop - start > timedelta(days=1):
                progressbar = tqdm
            else:
                progressbar = iter

        if progressbar is False:
            progressbar = iter

        progressbar = cast(Callable[[Iterable], Iterable], progressbar)

        other_params = ""

        if isinstance(icao24, str):
            other_params += "and icao24='{}' ".format(icao24.lower())

        elif isinstance(icao24, Iterable):
            icao24 = ",".join("'{}'".format(c.lower()) for c in icao24)
            other_params += "and icao24 in ({}) ".format(icao24)

        if isinstance(callsign, str):
            if callsign.find("%") > 0 or callsign.find("_") > 0:
                other_params += "and callsign ilike '{}' ".format(callsign)
            else:
                other_params += "and callsign='{:<8s}' ".format(callsign)

        elif isinstance(callsign, Iterable):
            callsign = ",".join("'{:<8s}'".format(c) for c in callsign)
            other_params += "and callsign in ({}) ".format(callsign)

        if departure_airport is not None:
            other_params += (
                f"and firstseen >= {start.timestamp()} and "
                f"firstseen < {stop.timestamp()} "
            )
        else:
            other_params += (
                f"and lastseen >= {start.timestamp()} and "
                f"lastseen < {stop.timestamp()} "
            )

        if airport:
            other_params += (
                f"and (estarrivalairport = '{airport}' or "
                f"estdepartureairport = '{airport}') "
            )
            if departure_airport is not None or arrival_airport is not None:
                raise RuntimeError(
                    "airport may not be set if "
                    "either arrival_airport or departure_airport is set"
                )
        else:
            if departure_airport:
                other_params += (
                    f"and estdepartureairport = '{departure_airport}' "
                )
            if arrival_airport:
                other_params += f"and estarrivalairport = '{arrival_airport}' "

        cumul = []
        sequence = list(split_times(start, stop, timedelta(days=1)))

        if limit is not None:
            other_params += f"limit {limit}"

        for bt, at, before_day, after_day in progressbar(sequence):

            logging.info(
                f"Sending request between time {bt} and {at} "
                f"and day {before_day} and {after_day}"
            )

            request = query_str.format(
                columns=columns,
                before_day=before_day.timestamp(),
                after_day=after_day.timestamp(),
                other_params=other_params,
            )

            df = self._impala(request, columns=columns, cached=cached)

            if df is None:
                continue

            df = self._format_dataframe(df)

            cumul.append(df)

        if len(cumul) == 0:
            return None

        df = (
            pd.concat(cumul, sort=True)
            .sort_values(
                "firstseen" if departure_airport is not None else "lastseen"
            )
            .rename(
                columns=dict(
                    estarrivalairport="destination",
                    estdepartureairport="origin",
                )
            )
        )

        return df

    def history(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        *args,  # more reasonable to be explicit about arguments
        date_delta: timedelta = timedelta(hours=1),  # noqa: B008
        return_flight: bool = False,
        callsign: Union[None, str, Iterable[str]] = None,
        icao24: Union[None, str, Iterable[str]] = None,
        serials: Union[None, int, Iterable[int]] = None,
        bounds: Union[
            BaseGeometry, Tuple[float, float, float, float], None
        ] = None,
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        airport: Optional[str] = None,
        count: bool = False,
        cached: bool = True,
        limit: Optional[int] = None,
        other_tables: str = "",
        other_params: str = "",
        nautical_units: bool = True,
        progressbar: Union[bool, Callable[[Iterable], Iterable]] = True,
    ) -> Optional[Union[Traffic, Flight]]:

        """Get Traffic from the OpenSky Impala shell.

        You may pass requests based on time ranges, callsigns, aircraft, areas,
        serial numbers for receivers, or airports of departure or arrival.

        The method builds appropriate SQL requests, caches results and formats
        data into a proper pandas DataFrame. Requests are split by hour (by
        default) in case the connection fails.

        Args:
            - **start**: a string (default to UTC), epoch or datetime (native
              Python or pandas)
            - **stop** (optional): a string (default to UTC), epoch or datetime
              (native Python or pandas), *by default, one day after start*
            - **date_delta** (optional): a timedelta representing how to split
              the requests, *by default: per hour*
            - **return_flight** (boolean, default: False): returns a Flight
              instead of a Traffic structure if switched to True

            More arguments to filter resulting data:

            - **callsign** (optional): a string or a list of strings (wildcards
              accepted, _ for any character, % for any sequence of characters);
            - **icao24** (optional): a string or a list of strings identifying i
              the transponder code of the aircraft;
            - **serials** (optional): an integer or a list of integers
              identifying the sensors receiving the data;
            - **bounds** (optional), sets a geographical footprint. Either
              an **airspace or shapely shape** (requires the bounds attribute);
              or a **tuple of float** (west, south, east, north);

            **Airports**

            The following options build more complicated requests by merging
            information from two tables in the Impala database, resp.
            `state_vectors_data4` and `flights_data4`.

            - **departure_airport** (optional): a string for the ICAO
              identifier of the airport. Selects flights departing from the
              airport between the two timestamps;
            - **arrival_airport** (optional): a string for the ICAO identifier
              of the airport. Selects flights arriving at the airport between
              the two timestamps;
            - **airport** (optional): a string for the ICAO identifier of the
              airport. Selects flights departing from or arriving at the
              airport between the two timestamps;

            .. warning::

                - See `opensky.flightlist
                  <#traffic.data.adsb.opensky_impala.Impala.flightlist>`__ if
                  you do not need any trajectory information.
                - If both departure_airport and arrival_airport are set,
                  requested timestamps match the arrival time;
                - If airport is set, departure_airport and
                  arrival_airport cannot be specified (a RuntimeException is
                  raised).

            **Useful options for debug**

            - **count** (boolean, default: False): add a column stating how
              many sensors received each record;
            - **nautical_units** (boolean, default: True): convert data stored
              in Impala to standard nautical units (ft, ft/min, knots).
            - **cached** (boolean, default: True): switch to False to force a
              new request to the database regardless of the cached files;
              delete previous cache files;
            - **limit** (optional, int): maximum number of records requested
              LIMIT keyword in SQL.

        """

        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        # default obvious parameter
        where_clause = "where"

        if progressbar is True:
            if stop - start > date_delta:
                progressbar = tqdm
            else:
                progressbar = iter

        if progressbar is False:
            progressbar = iter

        progressbar = cast(Callable[[Iterable], Iterable], progressbar)

        airports_params = [airport, departure_airport, arrival_airport]
        count_airports_params = sum(x is not None for x in airports_params)

        if isinstance(serials, Iterable):
            other_tables += ", state_vectors_data4.serials s "
            other_params += "and s.ITEM in {} ".format(tuple(serials))
        elif isinstance(serials, int):
            other_tables += ", state_vectors_data4.serials s "
            other_params += "and s.ITEM = {} ".format(serials)

        if isinstance(icao24, str):
            other_params += "and {}icao24='{}' ".format(
                "sv." if count_airports_params > 0 else "", icao24.lower()
            )

        elif isinstance(icao24, Iterable):
            icao24 = ",".join("'{}'".format(c.lower()) for c in icao24)
            other_params += "and {}icao24 in ({}) ".format(
                "sv." if count_airports_params > 0 else "", icao24
            )

        if isinstance(callsign, str):
            if (
                set(callsign)
                - set(string.ascii_letters)
                - set(string.digits)
                - set("%_")
            ):  # if regex like characters
                other_params += "and RTRIM({}callsign) REGEXP('{}') ".format(
                    "sv." if count_airports_params > 0 else "", callsign
                )
            elif callsign.find("%") > 0 or callsign.find("_") > 0:
                other_params += "and {}callsign ilike '{}' ".format(
                    "sv." if count_airports_params > 0 else "", callsign
                )
            else:
                other_params += "and {}callsign='{:<8s}' ".format(
                    "sv." if count_airports_params > 0 else "", callsign
                )

        elif isinstance(callsign, Iterable):
            callsign = ",".join("'{:<8s}'".format(c) for c in callsign)
            other_params += "and {}callsign in ({}) ".format(
                "sv." if count_airports_params > 0 else "", callsign
            )

        if bounds is not None:
            try:
                # thinking of shapely bounds attribute (in this order)
                # I just don't want to add the shapely dependency here
                west, south, east, north = bounds.bounds  # type: ignore
            except AttributeError:
                west, south, east, north = bounds

            other_params += "and lon>={} and lon<={} ".format(west, east)
            other_params += "and lat>={} and lat<={} ".format(south, north)

        day_min = round_time(start, how="before", by=timedelta(days=1))
        day_max = round_time(stop, how="after", by=timedelta(days=1))

        if count_airports_params > 0:
            where_clause = (
                "on sv.icao24 = est.e_icao24 and "
                "sv.callsign = est.e_callsign and "
                "est.firstseen <= sv.time and "
                "sv.time <= est.lastseen "
                "where"
            )

        if arrival_airport is not None and departure_airport is not None:
            if airport is not None:
                raise RuntimeError(
                    "airport may not be set if "
                    "either arrival_airport or departure_airport is set"
                )
            other_tables += (
                "as sv join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign as e_callsign, day from flights_data4 "
                "where estdepartureairport ='{departure_airport}' "
                "and estarrivalairport ='{arrival_airport}' "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                arrival_airport=arrival_airport,
                departure_airport=departure_airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        elif arrival_airport is not None:
            other_tables += (
                "as sv join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign as e_callsign, day from flights_data4 "
                "where estarrivalairport ='{arrival_airport}' "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                arrival_airport=arrival_airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        elif departure_airport is not None:
            other_tables += (
                "as sv join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign as e_callsign, day from flights_data4 "
                "where estdepartureairport ='{departure_airport}' "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                departure_airport=departure_airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        elif airport is not None:
            other_tables += (
                "as sv join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign as e_callsign, day from flights_data4 "
                "where (estdepartureairport ='{arrival_or_departure_airport}' "
                "or estarrivalairport = '{arrival_or_departure_airport}') "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                arrival_or_departure_airport=airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        cumul = []
        sequence = list(split_times(start, stop, date_delta))
        columns = ", ".join(self._impala_columns)
        parse_columns = ", ".join(self._impala_columns)

        if count_airports_params > 0:
            est_columns = [
                "firstseen",
                "estdepartureairport",
                "lastseen",
                "estarrivalairport",
                "day",
            ]
            columns = (
                ", ".join(f"sv.{field}" for field in self._impala_columns)
                + ", "
                + ", ".join(f"est.{field}" for field in est_columns)
            )
            parse_columns = ", ".join(
                self._impala_columns
                + ["firstseen", "origin", "lastseen", "destination", "day"]
            )

        if count is True:
            other_params += "group by " + columns
            columns = "count(*) as count, " + columns
            parse_columns = "count, " + parse_columns
            other_tables += ", state_vectors_data4.serials s"

        if limit is not None:
            other_params += f"limit {limit}"

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
                where_clause=where_clause,
            )

            df = self._impala(request, columns=parse_columns, cached=cached)

            if df is None:
                continue

            df = self._format_dataframe(df)
            df = self._format_history(df, nautical_units=nautical_units)

            if "last_position" in df.columns:
                if df.query("last_position == last_position").shape[0] == 0:
                    continue

            cumul.append(df)

        if len(cumul) == 0:
            return None

        df = pd.concat(cumul, sort=True).sort_values("timestamp")

        if count is True:
            df = df.assign(count=lambda df: df["count"].astype(int))

        if return_flight:
            return Flight(df)

        return Traffic(df)

    def rawdata(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        *args,  # more reasonable to be explicit about arguments
        table_name: Union[None, str, List[str]] = None,
        date_delta: timedelta = timedelta(hours=1),  # noqa: B008
        icao24: Union[None, str, Iterable[str]] = None,
        serials: Union[None, int, Iterable[int]] = None,
        bounds: Union[
            BaseGeometry, Tuple[float, float, float, float], None
        ] = None,
        callsign: Union[None, str, Iterable[str]] = None,
        departure_airport: Optional[str] = None,
        arrival_airport: Optional[str] = None,
        airport: Optional[str] = None,
        cached: bool = True,
        limit: Optional[int] = None,
        other_tables: str = "",
        other_columns: Union[None, str, List[str]] = None,
        other_params: str = "",
        progressbar: Union[bool, Callable[[Iterable], Iterable]] = True,
    ) -> Optional[RawData]:
        """Get raw message from the OpenSky Impala shell.

        You may pass requests based on time ranges, callsigns, aircraft, areas,
        serial numbers for receivers, or airports of departure or arrival.

        The method builds appropriate SQL requests, caches results and formats
        data into a proper pandas DataFrame. Requests are split by hour (by
        default) in case the connection fails.

        Args:
            - **start**: a string (default to UTC), epoch or datetime (native
              Python or pandas)
            - **stop** (optional): a string (default to UTC), epoch or datetime
              (native Python or pandas), *by default, one day after start*
            - **date_delta** (optional): a timedelta representing how to split
              the requests, *by default: per hour*

            More arguments to filter resulting data:

            - **callsign** (optional): a string or a list of strings (wildcards
              accepted, _ for any character, % for any sequence of characters);
            - **icao24** (optional): a string or a list of strings identifying i
              the transponder code of the aircraft;
            - **serials** (optional): an integer or a list of integers
              identifying the sensors receiving the data;
            - **bounds** (optional), sets a geographical footprint. Either
              an **airspace or shapely shape** (requires the bounds attribute);
              or a **tuple of float** (west, south, east, north);

            **Airports**

            The following options build more complicated requests by merging
            information from two tables in the Impala database, resp.
            `rollcall_replies_data4` and `flights_data4`.

            - **departure_airport** (optional): a string for the ICAO
              identifier of the airport. Selects flights departing from the
              airport between the two timestamps;
            - **arrival_airport** (optional): a string for the ICAO identifier
              of the airport. Selects flights arriving at the airport between
              the two timestamps;
            - **airport** (optional): a string for the ICAO identifier of the
              airport. Selects flights departing from or arriving at the
              airport between the two timestamps;

            .. warning::

                - If both departure_airport and arrival_airport are set,
                  requested timestamps match the arrival time;
                - If airport is set, departure_airport and
                  arrival_airport cannot be specified (a RuntimeException is
                  raised).
                - It is not possible at the moment to filter both on airports
                  and on geographical bounds (help welcome!).

            **Useful options for debug**

            - **cached** (boolean, default: True): switch to False to force a
              new request to the database regardless of the cached files;
              delete previous cache files;
            - **limit** (optional, int): maximum number of records requested
              LIMIT keyword in SQL.

        """

        if table_name is None:
            table_name = self._raw_tables

        if not isinstance(table_name, str):  # better than Iterable but not str
            return RawData.from_list(
                self.rawdata(
                    start,
                    stop,
                    table_name=table,
                    date_delta=date_delta,
                    icao24=icao24,
                    serials=serials,
                    bounds=bounds,
                    callsign=callsign,
                    departure_airport=departure_airport,
                    arrival_airport=arrival_airport,
                    airport=airport,
                    cached=cached,
                    limit=limit,
                    other_tables=other_tables,
                    other_columns=other_columns,
                    other_params=other_params,
                    progressbar=progressbar,
                )
                for table in table_name
            )

        _request = (
            "select {columns} from {table_name} {other_tables} "
            "{where_clause} hour>={before_hour} and hour<{after_hour} "
            "and {table_name}.mintime>={before_time} and "
            "{table_name}.mintime<{after_time} "
            "{other_params}"
        )

        columns = "mintime, maxtime, rawmsg, msgcount, icao24, hour"
        if other_columns is not None:
            if isinstance(other_columns, str):
                columns += f", {other_columns}"
            else:
                columns += ", " + ", ".join(other_columns)
        parse_columns = columns

        # default obvious parameter
        where_clause = "where"

        airports_params = [airport, departure_airport, arrival_airport]
        count_airports_params = sum(x is not None for x in airports_params)

        start = to_datetime(start)

        if table_name not in self._raw_tables:
            raise RuntimeError(f"{table_name} is not a valid table name")

        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(days=1)

        if progressbar is True:
            if stop - start > date_delta:
                progressbar = tqdm
            else:
                progressbar = iter

        if progressbar is False:
            progressbar = iter

        progressbar = cast(Callable[[Iterable], Iterable], progressbar)

        if isinstance(icao24, str):
            other_params += f"and {table_name}.icao24='{icao24.lower()}' "
        elif isinstance(icao24, Iterable):
            icao24 = ",".join("'{}'".format(c.lower()) for c in icao24)
            other_params += f"and {table_name}.icao24 in ({icao24}) "

        if isinstance(serials, Iterable):
            other_tables += f", {table_name}.sensors s "
            other_params += "and s.serial in {} ".format(tuple(serials))
            columns = "s.serial, s.mintime as time, " + columns
            parse_columns = "serial, time, " + parse_columns
        elif isinstance(serials, int):
            other_tables += f", {table_name}.sensors s "
            other_params += "and s.serial = {} ".format((serials))
            columns = "s.serial, s.mintime as time, " + columns
            parse_columns = "serial, time, " + parse_columns

        other_params += "and rawmsg is not null "

        day_min = round_time(start, how="before", by=timedelta(days=1))
        day_max = round_time(stop, how="after", by=timedelta(days=1))

        if (
            count_airports_params > 0
            or bounds is not None
            or callsign is not None
        ):

            where_clause = (
                f"on {table_name}.icao24 = est.e_icao24 and "
                f"est.firstseen <= {table_name}.mintime and "
                f"{table_name}.mintime <= est.lastseen "
                "where"
            )
        if callsign is not None:
            if count_airports_params > 0 or bounds is not None:
                raise RuntimeError(
                    "Either callsign, bounds or airport are "
                    "supported at the moment."
                )
            if isinstance(callsign, str):
                if callsign.find("%") > 0 or callsign.find("_") > 0:
                    callsigns = "and callsign ilike '{}' ".format(callsign)
                else:
                    callsigns = "and callsign='{:<8s}' ".format(callsign)

            elif isinstance(callsign, Iterable):
                callsign = ",".join("'{:<8s}'".format(c) for c in callsign)
                callsigns = "and callsign in ({}) ".format(callsign)

            other_tables += (
                "join (select min(time) as firstseen, max(time) as lastseen, "
                "icao24  as e_icao24 from state_vectors_data4 "
                "where hour>={before_hour} and hour<{after_hour} and "
                f"time>={start.timestamp()} and time<{stop.timestamp()} "
                f"{callsigns}"
                "group by icao24) as est "
            )

        elif bounds is not None:
            if count_airports_params > 0:
                raise RuntimeError(
                    "Either bounds or airport are supported at the moment."
                )
            try:
                # thinking of shapely bounds attribute (in this order)
                # I just don't want to add the shapely dependency here
                west, south, east, north = bounds.bounds  # type: ignore
            except AttributeError:
                west, south, east, north = bounds

            other_tables += (
                "join (select min(time) as firstseen, max(time) as lastseen, "
                "icao24 as e_icao24 from state_vectors_data4 "
                "where hour>={before_hour} and hour<{after_hour} and "
                f"time>={start.timestamp()} and time<{stop.timestamp()} and "
                f"lon>={west} and lon<={east} and "
                f"lat>={south} and lat<={north} "
                "group by icao24) as est "
            )

        elif arrival_airport is not None and departure_airport is not None:
            if airport is not None:
                raise RuntimeError(
                    "airport may not be set if "
                    "either arrival_airport or departure_airport is set"
                )
            other_tables += (
                "join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign, day from flights_data4 "
                "where estdepartureairport ='{departure_airport}' "
                "and estarrivalairport ='{arrival_airport}' "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                arrival_airport=arrival_airport,
                departure_airport=departure_airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        elif arrival_airport is not None:
            other_tables += (
                "join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign, day from flights_data4 "
                "where estarrivalairport ='{arrival_airport}' "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                arrival_airport=arrival_airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        elif departure_airport is not None:
            other_tables += (
                "join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign, day from flights_data4 "
                "where estdepartureairport ='{departure_airport}' "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                departure_airport=departure_airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        elif airport is not None:
            other_tables += (
                "join (select icao24 as e_icao24, firstseen, "
                "estdepartureairport, lastseen, estarrivalairport, "
                "callsign, day from flights_data4 "
                "where (estdepartureairport ='{arrival_or_departure_airport}' "
                "or estarrivalairport = '{arrival_or_departure_airport}') "
                "and ({day_min:.0f} <= day and day <= {day_max:.0f})) as est"
            ).format(
                arrival_or_departure_airport=airport,
                day_min=day_min.timestamp(),
                day_max=day_max.timestamp(),
            )

        fst_columns = [field.strip() for field in columns.split(",")]

        if count_airports_params > 1:
            est_columns = [
                "firstseen",
                "estdepartureairport",
                "lastseen",
                "estarrivalairport",
                "day",
            ]
            columns = (
                ", ".join(f"{table_name}.{field}" for field in fst_columns)
                + ", "
                + ", ".join(f"est.{field}" for field in est_columns)
            )
            parse_columns = ", ".join(
                fst_columns
                + ["firstseen", "origin", "lastseen", "destination", "day"]
            )
        if bounds is not None:
            columns = (
                ", ".join(f"{table_name}.{field}" for field in fst_columns)
                + ", "
                + ", ".join(
                    f"est.{field}"
                    for field in ["firstseen", "lastseen", "e_icao24"]
                )
            )
            parse_columns = ", ".join(
                fst_columns + ["firstseen", "lastseen", "icao24_2"]
            )

        sequence = list(split_times(start, stop, date_delta))
        cumul = []

        if limit is not None:
            other_params += f"limit {limit}"

        for bt, at, bh, ah in progressbar(sequence):

            logging.info(
                f"Sending request between time {bt} and {at} "
                f"and hour {bh} and {ah}"
            )

            if "{before_hour}" in other_tables:
                _other_tables = other_tables.format(
                    before_hour=bh.timestamp(), after_hour=ah.timestamp()
                )
            else:
                _other_tables = other_tables

            request = _request.format(
                columns=columns,
                table_name=table_name,
                before_time=int(bt.timestamp()),
                after_time=int(at.timestamp()),
                before_hour=bh.timestamp(),
                after_hour=ah.timestamp(),
                other_tables=_other_tables,
                other_params=other_params,
                where_clause=where_clause,
            )

            df = self._impala(request, columns=parse_columns, cached=cached)

            if df is None:
                continue

            df = self._format_dataframe(df)

            cumul.append(df)

        if len(cumul) == 0:
            return None

        return RawData(pd.concat(cumul).sort_values("mintime"))

    def extended(self, *args, **kwargs) -> Optional[RawData]:
        return self.rawdata(
            table_name="rollcall_replies_data4", *args, **kwargs
        )


Impala.extended.__doc__ = Impala.rawdata.__doc__
Impala.extended.__annotations__ = {
    key: value
    for (key, value) in Impala.rawdata.__annotations__.items()
    if key != "table_name"
}

# below this line is only helpful references
# ------------------------------------------
# [hadoop-1:21000] > describe rollcall_replies_data4;
# +----------------------+-------------------+---------+
# | name                 | type              | comment |
# +----------------------+-------------------+---------+
# | sensors              | array<struct<     |         |
# |                      |   serial:int,     |         |
# |                      |   mintime:double, |         |
# |                      |   maxtime:double  |         |
# |                      | >>                |         |
# | rawmsg               | string            |         |
# | mintime              | double            |         |
# | maxtime              | double            |         |
# | msgcount             | bigint            |         |
# | icao24               | string            |         |
# | message              | string            |         |
# | isid                 | boolean           |         |
# | flightstatus         | tinyint           |         |
# | downlinkrequest      | tinyint           |         |
# | utilitymsg           | tinyint           |         |
# | interrogatorid       | tinyint           |         |
# | identifierdesignator | tinyint           |         |
# | valuecode            | smallint          |         |
# | altitude             | double            |         |
# | identity             | string            |         |
# | hour                 | int               |         |
# +----------------------+-------------------+---------+
