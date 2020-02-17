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

    @staticmethod
    def position_extraction(os_pos, lat_ref=None, lon_ref=None) -> pd.DataFrame:
        if os_pos.empty:
            return os_pos

        # pandas option to get rid of SettingWithCopyWarning
        pd.options.mode.chained_assignment = None

        if os_pos["mintime"].dtypes == np.float64:
            os_pos["mintime"] = pd.to_datetime(os_pos["mintime"], unit="s")

        os_pos["oe_flag"] = os_pos["rawmsg"].apply(adsb.oe_flag)
        os_pos["altitude"] = os_pos["rawmsg"].apply(adsb.altitude)
        os_pos["icao24"] = os_pos["rawmsg"].str[2:8]
        os_pos.sort_values(by=["icao24", "mintime"], inplace=True)
        os_pos["inv"] = np.nan
        os_pos["inv_time"] = np.datetime64("NaT")

        # Loop on the last 10 messages to find a message of the other parity.
        for j in range(10, 0, -1):

            # inv is the other parity message found. nan if none found.
            os_pos["inv"] = np.where(
                (os_pos.shift(j)["icao24"] == os_pos["icao24"])
                & (os_pos["oe_flag"].shift(j) != os_pos["oe_flag"])
                & (
                    os_pos["mintime"].shift(j)
                    > os_pos["mintime"] - pd.Timedelta(seconds=10)
                ),
                os_pos.shift(j)["rawmsg"],
                os_pos.inv,
            )

            # inv_time is the ts associated to the message. nan if none found.
            os_pos["inv_time"] = np.where(
                (os_pos.shift(j)["icao24"] == os_pos["icao24"])
                & (os_pos["oe_flag"].shift(j) != os_pos["oe_flag"])
                & (
                    os_pos["mintime"].shift(j)
                    > os_pos["mintime"] - pd.Timedelta(seconds=10)
                ),
                os_pos.shift(j)["mintime"],
                os_pos.inv_time,
            )

        os_pos.inv_time = os_pos.inv_time.dt.tz_localize("UTC")

        # apply adsb.position. TODO : calc with matrix. Ask Junzi.
        pos_tmp = os_pos.loc[~os_pos["inv"].isnull(), :].apply(
            lambda row: adsb.position(
                row.rawmsg, row.inv, row.mintime, row.inv_time
            ),
            axis=1,
        )
        pos = pd.DataFrame(
            pos_tmp.tolist(),
            columns=["latitude", "longitude"],
            index=pos_tmp.index,
        )
        os_pos.drop(["inv", "inv_time", "oe_flag"], axis=1, inplace=True)
        result = pd.concat([os_pos, pos], axis=1, join="inner")
        return result

    @staticmethod
    def velocity_extraction(os_vel) -> pd.DataFrame:
        if os_vel.empty:
            return os_vel
        if os_vel["mintime"].dtypes == np.float64:
            os_vel["mintime"] = pd.to_datetime(os_vel["mintime"], unit="s")
        os_vel["speed_temp"] = os_vel["rawmsg"].apply(adsb.velocity)
        os_vel[["speed", "heading", "vertical_rate", "type"]] = pd.DataFrame(
            os_vel["speed_temp"].tolist(), index=os_vel.index
        )
        os_vel.drop(["speed_temp", "type"], axis=1, inplace=True)
        return os_vel

    @staticmethod
    def identification_extraction(os_ide) -> pd.DataFrame:
        if os_ide.empty:
            return os_ide

        if os_ide["mintime"].dtypes == np.float64:
            os_ide["mintime"] = pd.to_datetime(os_ide["mintime"], unit="s")
        os_ide["callsign"] = os_ide["rawmsg"].apply(callsign)
        return os_ide

    def feature_extraction(self, lat_ref=None, lon_ref=None) -> "RawData":
        if "msg_type" not in self.data.columns:
            self.get_type()

        ide = RawData.identification_extraction(
            self.data.loc[self.data["msg_type"] == 1]
        )
        vel = RawData.velocity_extraction(
            self.data.loc[self.data["msg_type"] == 4]
        )
        pos = RawData.position_extraction(
            self.data.loc[self.data["msg_type"] == 3], lat_ref, lon_ref
        )
        result = pd.concat(
            [ide, vel, pos], axis=0, sort=False, ignore_index=True
        )
        result = result.sort_values(by=["mintime", "icao24"])
        return RawData(result)

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
