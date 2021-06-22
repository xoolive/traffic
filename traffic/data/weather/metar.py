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

    def get(self, start: Optional[timelike]):

        timestamp = to_datetime(start) if start is not None else datetime.now()

        c = requests.get(
            "http://weather.uwyo.edu/cgi-bin/wyowx.fcgi?TYPE=metar&"
            f"DATE={timestamp:%Y%m%d}&STATION={self.airport}"
        )
        c.raise_for_status()

        list_ = (
            bs4.BeautifulSoup(c.content).find("pre").text.strip().split("\n")
        )
        df = pd.DataFrame.from_records([vars(Metar.Metar(m)) for m in list_])

        return (
            df.drop(columns=[c for c in df.columns if c.startswith("_")])
            .assign(time=lambda df: df.time.dt.tz_localize("utc"))
            .dropna(axis=1)
            .iloc[::-1]
        )
