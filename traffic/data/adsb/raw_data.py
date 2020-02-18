from typing import Callable, Dict, Iterable, Type, TypeVar

import numpy as np
import pandas as pd
from pyModeS import adsb
from pyModeS.decoder.bds.bds08 import callsign

from ...core.mixins import DataFrameMixin

T = TypeVar("T", bound="RawData")


def encode_time_dump1090(times: pd.Series) -> pd.Series:
    ref_time = times.iloc[0]
    rel_times = times - ref_time
    rel_times = rel_times * 12e6
    rel_times = rel_times.apply(lambda row: hex(int(row))[2:].zfill(12))
    return rel_times


encode_time: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "dump1090": encode_time_dump1090
}


class RawData(DataFrameMixin):
    def __add__(self: T, other: T) -> T:
        return self.__class__.from_list([self, other])

    @classmethod
    def from_list(cls: Type[T], elts: Iterable[T]) -> T:
        res = cls(pd.concat(list(x.data for x in elts), sort=False))
        return res.sort_values("mintime")

    def get_type(self, inplace=False):
        def get_typecode(msg):
            tc = adsb.typecode(msg)
            if 9 <= tc <= 18:
                return 3
            elif tc == 19:
                return 4
            elif 1 <= tc <= 4:
                return 1
            else:
                return np.nan

        if inplace:
            self.data["msg_type"] = self.data.rawmsg.apply(get_typecode)
            self.data.dropna(subset=["msg_type"], inplace=True)
        else:
            return self.data.rawmsg.apply(get_typecode)

    def to_beast(self, time_fmt: str = "dump1090") -> pd.Series:
        df_beast = self.data[["mintime", "rawmsg"]].copy()
        if isinstance(df_beast.mintime.iloc[0], pd.datetime):
            df_beast.mintime = df_beast.mintime.astype(np.int64) / 10 ** 9
        encoder = encode_time.get(time_fmt, encode_time_dump1090)
        df_beast["time"] = encoder(df_beast.mintime)
        df_beast["message"] = "@" + df_beast.time + df_beast.rawmsg
        return df_beast["message"]
