from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

from metar import Metar

import pandas as pd

from ...core.structure import Airport
from ...core.time import timelike, to_datetime
from .. import client

list_ap = Union[List[str], List[Airport]]


class METAR:  # coverage: ignore
    def __init__(self, airport: Union[str, Airport, list_ap]):
        if isinstance(airport, str):
            self.airports = [airport]
        elif isinstance(airport, Airport):
            self.airports = [airport.icao]
        else:
            self.airports = []
            for ap in airport:
                if isinstance(ap, str):
                    self.airports.append(ap)
                else:
                    self.airports.append(ap.icao)

    def get(
        self, start: Optional[timelike] = None, stop: Optional[timelike] = None
    ) -> pd.DataFrame:
        """Retrieve METAR infos.

        Parameters
        ----------
        start : Optional[timelike]
            Time for which METAR info are retrieved. If no hour is specified,
            data are retrieved for the day. If no time is specified, the
            current METAR information are retrieved.
        stop : Optional[timelike]
            Time until which METAR info are retrieved.
            If no stop time is specified, data are retrieved for 24 hours.

        Returns
        -------
        pd.DataFrame
            DataFrame containing METAR information.
        """
        start = (
            to_datetime(start)
            if start is not None
            else datetime.now(tz=timezone.utc)
        )
        stop = (
            to_datetime(stop)
            if stop is not None
            else start + timedelta(hours=24)
        )
        if stop == start:
            stop += timedelta(hours=1)

        str_ap = ""
        for ap in self.airports:
            str_ap += f"station={ap}&"
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        url += str_ap
        url += (
            f"data=metar&year1={start.year}&"
            f"month1={start.month}&"
            f"day1={start.day}&"
            f"hour1={start.hour}&"
            f"year2={stop.year}&"
            f"month2={stop.month}&"
            f"day2={stop.day}&"
            f"hour2={stop.hour}&"
            f"tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&"
            f"missing=M&trace=T&direct=no&report_type=3&report_type=4"
        )
        c = client.get(url)
        c.raise_for_status()
        list_ = c.content.decode("utf-8").strip().split("\n")
        df_metar = pd.DataFrame.from_records(
            [
                vars(
                    Metar.Metar(
                        m.split(",")[-1],
                        month=datetime.strptime(
                            m.split(",")[1], "%Y-%m-%d %H:%M"
                        )
                        .replace(tzinfo=timezone.utc)
                        .month,
                        year=datetime.strptime(
                            m.split(",")[1], "%Y-%m-%d %H:%M"
                        )
                        .replace(tzinfo=timezone.utc)
                        .year,
                        strict=False,
                        utcdelta=timedelta(hours=0),
                    )
                )
                for m in list_[1:]
            ]
        )
        df_metar["time"] = df_metar["time"].dt.tz_localize("utc")
        df_metar["airport"] = (
            df_metar["code"].str.split(" ").apply(lambda x: x[0])
        )
        return df_metar.drop(
            columns=[c for c in df_metar.columns if c.startswith("_")]
        )
