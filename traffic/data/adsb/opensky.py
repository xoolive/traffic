import hashlib
import logging
import re
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple, Union, cast

import pandas as pd
import paramiko
import requests

from ...core import Flight, Traffic
from ...core.time import round_time, split_times, timelike, to_datetime
from ..basic.airport import Airport


class OpenSky(object):

    columns = ["icao24", "callsign", "origin_country", "last_position",
               "timestamp", "longitude", "latitude", "altitude", "onground",
               "ground_speed", "track", "vertical_rate", "sensors",
               "baro_altitude", "squawk", "spi", "position_source"]

    basic_request = ("select * from state_vectors_data4 {other_columns} "
                     "where hour>={before_hour} and hour<{after_hour} "
                     "and time>={before_time} and time<{after_time} "
                     "{other_where}")

    airport_request = ("select icao24, callsign from state_vectors_data4 "
                       "where lat<={airport_latmax} and lat>={airport_latmin} "
                       "and lon<={airport_lonmax} and lon>={airport_lonmin} "
                       "and baroaltitude<=1000 "
                       "and hour>={before_hour} and hour<{after_hour} "
                       "and time>={before_time} and time<{after_time} "
                       "group by icao24, callsign")

    sensor_request = ("select icao24, callsign, s.ITEM as serial,"
                      "count(*) as count from "
                      "state_vectors_data4, state_vectors_data4.serials s "
                      "where hour>={before_hour} and hour<{after_hour} "
                      "{other_where} group by icao24, callsign, s.ITEM")

    shell: Optional[Any]  # the connection to the Impala shell

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
        for file in self.cache_dir.glob('*'):
            file.unlink()

    @staticmethod
    def _format_dataframe(df: pd.DataFrame,
                          nautical_units=True) -> pd.DataFrame:

        df.callsign = df.callsign.str.strip()

        if nautical_units:
            df.altitude = df.altitude / 0.3048
            df.baro_altitude = df.baro_altitude / 0.3048
            df.ground_speed = df.ground_speed / 1852 * 3600
            df.vertical_rate = df.vertical_rate / 0.3048 * 60

        df.timestamp = df.timestamp.apply(datetime.fromtimestamp)
        # warning is raised here: SettingWithCopyWarning:
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # Try using .loc[row_indexer,col_indexer] = value instead
        pd.options.mode.chained_assignment = None
        df = df.loc[df.last_position.notnull()]  # do we really miss much?
        df.last_position = df.last_position.apply(datetime.fromtimestamp)
        df = df.sort_values('timestamp')

        return df

    def _connect(self) -> None:
        if self.username == "" or self.password == "":
            raise RuntimeError("This method requires authentication.")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect("data.opensky-network.org", port=2230,
                       username=self.username, password=self.password)
        self.shell = client.invoke_shell()
        self.connected = True
        total = ""
        while len(total) == 0 or total[-19:] != "[hadoop-1:21000] > ":
            b = self.shell.recv(256)
            total += b.decode()

    def _impala(self, request: str) -> Optional[pd.DataFrame]:

        digest = hashlib.md5(request.encode('utf8')).hexdigest()
        cachename = self.cache_dir / digest

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
            with cachename.open('w') as fh:
                fh.write(total)

        logging.info("Reading request in cache {}".format(cachename))
        with cachename.open('r') as fh:
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

    def history(self, before: timelike, after: Optional[timelike]=None,
                date_delta: timedelta=timedelta(hours=1),
                callsign: Optional[Union[str, Iterable[str]]]=None,
                serials: Optional[Iterable[int]]=None,
                bounds: Optional[Tuple[float, float, float, float]]=None,
                other_columns: str="", other_where: str="",
                progressbar: Callable[[Iterable], Iterable]=iter
                ) -> Optional[Union[Traffic, Flight]]:

        return_flight = False
        before = to_datetime(before)
        if after is not None:
            after = to_datetime(after)
        else:
            after = before + timedelta(days=1)

        if isinstance(serials, Iterable):
            other_columns += ", state_vectors_data4.serials s "
            other_where += "and s.ITEM in {} ".format(tuple(serials))

        if isinstance(callsign, str):
            other_where += "and callsign='{:<8s}' ".format(callsign)
            return_flight = True

        elif isinstance(callsign, Iterable):
            callsign = ",".join("'{:<8s}'".format(c) for c in callsign)
            other_where += "and callsign in ({}) ".format(callsign)

        if bounds is not None:
            try:
                # thinking of shapely bounds attribute (in this order)
                # I just don't want to add the shapely dependency here
                west, south, east, north = bounds.bounds  # type: ignore
            except AttributeError:
                west, south, east, north = bounds

            other_where += "and lon>={} and lon<={} ".format(west, east)
            other_where += "and lat>={} and lat<={} ".format(south, north)

        cumul = []
        sequence = list(split_times(before, after, date_delta))

        for bt, at, bh, ah in progressbar(sequence):

            logging.info(
                f"Sending request between time {bt} and {at} "
                f"and hour {bh} and {ah}")

            request = self.basic_request.format(
                before_time=bt.timestamp(), after_time=at.timestamp(),
                before_hour=bh.timestamp(), after_hour=ah.timestamp(),
                other_columns=other_columns, other_where=other_where)

            df = self._impala(request)

            if df is None:
                continue

            df = df.drop(['lastcontact', ], axis=1)

            # TODO option for callsign/icao24
            # TODO remove serials as well (option)
            df = df.drop_duplicates(('callsign', 'time'))
            if df.lat.dtype == object:
                df = df[df.lat != 'lat']  # header is regularly repeated

            # restore all types
            for column_name in ['lat', 'lon', 'velocity', 'heading',
                                'geoaltitude', 'baroaltitude', 'vertrate',
                                'time', 'lastposupdate']:
                df[column_name] = df[column_name].astype(float)

            df.icao24 = df.icao24.apply(
                lambda x: "{:0>6}".format(hex(int(str(x), 16))[2:]))

            if df.onground.dtype != bool:
                df.onground = (df.onground == 'true')
                df.alert = (df.alert == 'true')
                df.spi = (df.spi == 'true')

            # better (to me) formalism about columns
            df = df.rename(columns={'lat': 'latitude',
                                    'lon': 'longitude',
                                    'heading': 'track',
                                    'velocity': 'ground_speed',
                                    'vertrate': 'vertical_rate',
                                    'baroaltitude': 'baro_altitude',
                                    'geoaltitude': 'altitude',
                                    'time': 'timestamp',
                                    'lastposupdate': 'last_position'})

            df = self._format_dataframe(df)
            cumul.append(df)

        if len(cumul) == 0:
            return None

        if return_flight:
            return Flight(pd.concat(cumul))

        return Traffic(pd.concat(cumul))

    def sensors(
            self, before: timelike, after: timelike,
            bounds: Tuple[float, float, float, float]
    ) -> Optional[pd.DataFrame]:

        before_hour = round_time(before, "before")
        after_hour = round_time(after, "after")

        try:
            # thinking of shapely bounds attribute (in this order)
            # I just don't want to add the shapely dependency here
            west, south, east, north = bounds.bounds  # type: ignore
        except AttributeError:
            west, south, east, north = bounds

        other_where = "and lon>={} and lon<={} ".format(west, east)
        other_where += "and lat>={} and lat<={} ".format(south, north)

        query = self.sensor_request.format(before_hour=before_hour,
                                           after_hour=after_hour,
                                           other_where=other_where)

        logging.info(f"Sending request: {query}")
        df = self._impala(query)
        if df is None:
            return None

        df = df[df['count'] != 'count']
        df['count'] = df['count'].astype(int)

        return df

    def online_aircraft(self, own=False) -> pd.DataFrame:
        what = "own" if (own and self.username != ""
                         and self.password != "") else "all"
        c = requests.get(f"https://opensky-network.org/api/states/{what}",
                         auth=self.auth)
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        r = pd.DataFrame.from_records(c.json()['states'], columns=self.columns)
        r = r.drop(['origin_country', 'spi', 'sensors'], axis=1)
        r = r.dropna()
        return self._format_dataframe(r, nautical_units=True)

    def at_airport(self, before: timelike, after: timelike,
                   airport: Airport) -> Optional[pd.DataFrame]:

        before = to_datetime(before)
        after = to_datetime(after)

        before_hour = round_time(before, how='before')
        after_hour = round_time(after, how='after')

        request = self.airport_request.format(
            before_time=before.timestamp(),
            after_time=after.timestamp(),
            before_hour=before_hour.timestamp(),
            after_hour=after_hour.timestamp(),
            airport_latmax=airport.lat + 0.1,
            airport_latmin=airport.lat - 0.1,
            airport_lonmax=airport.lon + 0.1,
            airport_lonmin=airport.lon - 0.1,)

        df = self._impala(request)

        return df
