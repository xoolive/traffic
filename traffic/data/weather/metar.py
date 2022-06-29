from datetime import datetime
from typing import Optional, Union

import bs4
import requests
from metar import Metar

import pandas as pd

from ...core.structure import Airport
from ...core.time import timelike, to_datetime


class METAR:  # coverage: ignore
    def __init__(self, airport: Union[str, Airport]):
        self.airport = airport if isinstance(airport, str) else airport.icao

    def get(self, start: Optional[timelike]) -> pd.Dataframe:
        """Retieve METAR infos.

        Parameters
        ----------
        start : Optional[timelike]
            Time for which METAR info are retrieved. If no hour is specified,
            data are retrieved for the day. If no time is specified, the
            current METAR information are retrieved.

        Returns
        -------
        pd.Dataframe
            DataFrame containing METAR information.
        """
        timestamp = to_datetime(start) if start is not None else datetime.now()
        if timestamp.hour != 0:
            c = requests.get(
                "http://weather.uwyo.edu/cgi-bin/wyowx.fcgi?TYPE=metar&"
                f"DATE={timestamp:%Y%m%d}"
                f"&HOUR={timestamp.hour+1}"
                f"&STATION={self.airport}"
            )
        else:
            c = requests.get(
                "http://weather.uwyo.edu/cgi-bin/wyowx.fcgi?TYPE=metar&"
                f"DATE={timestamp:%Y%m%d}"
                f"&STATION={self.airport}"
            )
        c.raise_for_status()

        list_ = (
            bs4.BeautifulSoup(c.content).find("pre").text.strip().split("\n")
        )
        df = pd.DataFrame.from_records([vars(Metar.Metar(m)) for m in list_])
        df = df.drop_duplicates("time").query("_day == @timestamp.day")
        df["time"] = df["time"].apply(
            lambda dt: dt.replace(
                year=timestamp.year, month=timestamp.month
            ).tz_localize("utc")
        )
        if timestamp.hour != 0:
            df = df.loc[[(abs(df.time - timestamp)).idxmin()]]

        return (
            df.drop(columns=[c for c in df.columns if c.startswith("_")])
            .dropna(axis=1)
            .set_index("time")
            .sort_index()
        )
