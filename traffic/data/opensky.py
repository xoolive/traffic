import logging
from typing import Optional, Tuple

import pandas as pd
import requests

from ..core.time import round_time, timelike, to_datetime
from .airport import Airport
from .impala import ImpalaWrapper


class OpenSky(ImpalaWrapper):

    columns = ["icao24", "callsign", "origin_country", "last_position",
               "timestamp", "longitude", "latitude", "altitude", "onground",
               "ground_speed", "track", "vertical_rate", "sensors",
               "baro_altitude", "squawk", "spi", "position_source"]

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

    def __init__(self, username: str="", password: str="") -> None:
        super().__init__(username, password)
        if username == "" or password == "":
            self.auth = None
        else:
            self.auth = (username, password)

    def sensors(self, before: timelike, after: timelike,
                bounds: Tuple[float, float, float, float]) -> pd.DataFrame:

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

    def history(self, before: timelike, after: timelike,
                *args, **kwargs) -> Optional[pd.DataFrame]:

        before = to_datetime(before)
        after = to_datetime(after)
        return ImpalaWrapper.history(self, before, after, *args, **kwargs)
