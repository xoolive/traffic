import hashlib
import logging
import re
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

import pandas as pd
import paramiko
from tqdm import tqdm

from ..core.time import round_time


class ImpalaWrapper(object):

    basic_request = ("select * from state_vectors_data4 {other_columns} "
                     "where hour>={before_hour} and hour<{after_hour} "
                     "and time>={before_time} and time<{after_time} "
                     "{other_where}")

    cache_dir: Path

    def __init__(self, username: str, password: str) -> None:

        self.username = username
        self.password = password
        self.connected = False

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
        self.c = client.invoke_shell()
        self.connected = True
        total = ""
        while len(total) == 0 or total[-19:] != "[hadoop-1:21000] > ":
            b = self.c.recv(256)
            total += b.decode()

    def _impala(self, request: str) -> Optional[pd.DataFrame]:

        digest = hashlib.md5(request.encode('utf8')).hexdigest()
        cachename = self.cache_dir / digest

        if not cachename.exists():
            if not self.connected:
                self._connect()
            logging.info("Sending request: {}".format(request))
            self.c.send(request + ";\n")
            total = ""
            while len(total) == 0 or total[-19:] != "[hadoop-1:21000] > ":
                b = self.c.recv(256)
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

    def history(self, before: datetime, after: datetime,
                callsign: Optional[Union[str, Iterable[str]]]=None,
                serials: Optional[Iterable[int]]=None,
                bounds: Optional[Tuple[float, float, float, float]]=None,
                other_columns: str="",
                other_where: str="") -> Optional[pd.DataFrame]:

        before_hour = round_time(before, how='before')
        after_hour = round_time(after, how='after')

        if isinstance(serials, Iterable):
            other_columns += ", state_vectors_data4.serials s "
            other_where += "and s.ITEM in {} ".format(tuple(serials))

        if isinstance(callsign, str):
            other_where += "and callsign='{:<8s}' ".format(callsign)

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

        seq = np.arange(before_hour, after_hour + timedelta(hours=1),
                        timedelta(hours=1)).astype(datetime)
        cumul = []

        for before, after in tqdm(zip(seq, seq[1:]), total=len(seq)-1):

            request = self.basic_request.format(
                before_time=before.timestamp(),
                after_time=after.timestamp(),
                before_hour=before_hour.timestamp(),
                after_hour=after_hour.timestamp(),
                other_columns=other_columns,
                other_where=other_where)

            df = self._impala(request)

            if df is None:
                continue

            df = df.drop(['lastcontact', ], axis=1)
            # may be useful if several serials
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

        if len(cumul) > 0:
            return pd.concat(cumul)

        return None
