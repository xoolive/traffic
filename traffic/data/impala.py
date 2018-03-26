import hashlib
import re
import tempfile
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

import pandas as pd
import paramiko
from tqdm import tqdm


class ImpalaWrapper(object):

    basic_request = ("select * from state_vectors_data4 {other_columns}"
                     "where hour>={before_hour} and hour<{after_hour} "
                     "and time>={before_time} and time<{after_time} "
                    "{other_where}")

    def __init__(self, username: str="", password: str="") -> None:

        self.username = username
        self.password = password

        if username == "" or password == "":
            self.auth = None
        else:
            self.auth = (username, password)
        self.connected = False

        self.cache_dir = Path(tempfile.gettempdir()) / "cache_opensky"
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    @staticmethod
    def _round_time(dt: datetime, how: str='before',
                    date_delta: timedelta=timedelta(hours=1)) -> datetime:

        round_to = date_delta.total_seconds()
        seconds = (dt - dt.min).seconds

        if how == 'before':
            rounding = (seconds + round_to) // round_to * round_to
        elif how == 'after':
            rounding = seconds // round_to * round_to
        else:
            raise ValueError("parameter how must be `before` or `after`")

        return dt + timedelta(0, rounding - seconds, -dt.microsecond)

    @staticmethod
    def _format_dataframe(df: pd.DataFrame, nautical_units=True) -> pd.DataFrame:

        df.callsign = df.callsign.str.strip()

        if nautical_units:
            df.altitude = df.altitude / 0.3048
            df.baro_altitude = df.baro_altitude / 0.3048
            df.ground_speed = df.ground_speed / 1852 * 3600
            df.vertical_rate = df.vertical_rate / 0.3048 * 60

        df.timestamp = df.timestamp.apply(datetime.fromtimestamp)
        df.last_position = df.last_position.apply(datetime.fromtimestamp)
        df = df.loc[df.last_position.notnull()]  # do we really miss much?
        df = df.sort_values('timestamp')

        return df

    def _connect(self) -> None:
        if self.username == "" or self.password == "":
            raise NotImplementedError("This method requires authentication.")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect("data.opensky-network.org", port=2230,
                       username=self.username,
                       password=self.password)
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
            self.c.send(request + ";\n")
            total = ""
            while len(total) == 0 or total[-19:] != "[hadoop-1:21000] > ":
                b = self.c.recv(256)
                total += b.decode()
            with cachename.open('w') as fh:
                fh.write(total)

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
                serials=None,
                bounds: Optional[Tuple[float, float, float, float]]=None,
                other_columns: str="", other_where: str="") -> pd.DataFrame:

        before_hour = self._round_time(before, how='before')
        after_hour = self._round_time(after, how='after')

        if serials is not None:
            other_columns += ", state_vectors_data4.serials s "

        if bounds is not None:
            try:
                # thinking of shapely bounds attribute (in this order)
                # I just don't want to add the shapely dependency here
                west, south, east, north = bounds.bounds  # type: ignore
            except:
                west, south, east, north = bounds

        seq = np.arange(before_hour, after_hour + timedelta(hours=1),
                        timedelta(hours=1)).astype(datetime)
        cumul = []

        for before, after in tqdm(zip(seq, seq[1:]), total=len(seq)-1):

            request = self.basic_request.format(
                before_time=before.timestamp(),
                after_time=after.timestamp(),
                before_hour=before.timestamp(),
                after_hour=after.timestamp(),
                other_columns=other_columns,
                other_where=other_where)

            if isinstance(callsign, str):
                request += "and callsign='{:<8s}' ".format(callsign)

            elif isinstance(callsign, Iterable):
                callsign = ",".join("'{:<8s}'".format(c) for c in callsign)
                request += "and callsign in ({}) ".format(callsign)

            if bounds is not None:
                request += "and lon>={} and lon<={} ".format(west, east)
                request += "and lat>={} and lat<={} ".format(south, north)

            df = self._impala(request)

            if df is None:
                continue

            df = df.drop(['lastcontact', ], axis=1)
            # may be useful if several serials
            df = df.drop_duplicates(('callsign', 'time'))
            df = df[df.lat != 'lat']  # header is regularly repeated

            # restore all types
            for column_name in ['lat', 'lon', 'velocity', 'heading', 'heading',
                                'geoaltitude', 'baroaltitude', 'vertrate',
                                'time', 'lastposupdate']:
                df[column_name] = df[column_name].astype(float)

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
