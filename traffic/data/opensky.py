from typing import Optional

import pandas as pd

from ..core.time import timelike, to_datetime
from .impala import ImpalaWrapper


class OpenSky(ImpalaWrapper):

    columns = ["icao24", "callsign", "origin_country", "last_position",
               "timestamp", "longitude", "latitude", "altitude", "onground",
               "ground_speed", "track", "vertical_rate", "sensors",
               "baro_altitude", "squawk", "spi", "position_source"]

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
        return self.format_dataframe(r, nautical_units = True)

    def history(self, before: timelike, after: timelike,
                *args, **kwargs) -> Optional[pd.DataFrame]:

        before = to_datetime(before)
        after = to_datetime(after)
        return ImpalaWrapper.history(self, before, after, *args, **kwargs)

