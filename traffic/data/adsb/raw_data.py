import numpy as np
import pandas as pd

from pyModeS import adsb
from pyModeS.decoder.bds.bds08 import callsign

from ...core.mixins import DataFrameMixin


class RawData(DataFrameMixin):
    @property
    def _constructor(self):
        return RawData

    @staticmethod
    def raw_cleaner(data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def position_extraction(os_pos, lat_ref=None,
                            lon_ref=None) -> pd.DataFrame:
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
        os_pos["inv_time"] = np.datetime64('NaT')

        # Loop on the last 10 messages to find a message of the other parity.
        for j in range(10, 0, -1):

            # inv is the other parity message found. nan if none found.
            os_pos["inv"] = np.where(
                (os_pos.shift(j)["icao24"] == os_pos["icao24"])
                & (os_pos["oe_flag"].shift(j) != os_pos["oe_flag"])
                & (os_pos["mintime"].shift(j) >
                   os_pos["mintime"] - pd.Timedelta(seconds=10)),
                os_pos.shift(j)["rawmsg"],
                os_pos.inv,
            )

            # inv_time is the ts associated to the message. nan if none found.
            os_pos["inv_time"] = np.where(
                (os_pos.shift(j)["icao24"] == os_pos["icao24"])
                & (os_pos["oe_flag"].shift(j) != os_pos["oe_flag"])
                & (os_pos["mintime"].shift(j) >
                   os_pos["mintime"] - pd.Timedelta(seconds=10)),
                os_pos.shift(j)["mintime"],
                os_pos.inv_time,
            )

        os_pos.inv_time = os_pos.inv_time.dt.tz_localize('UTC')

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

    def feature_extraction(self, lat_ref=None, lon_ref=None) -> pd.DataFrame:
        if 'msg_type' not in self.data.columns:
            self.get_type()

        ide = RawData.identification_extraction(
            self.data.loc[self.data["msg_type"] == 1]
        )
        vel = RawData.velocity_extraction(self.data.loc[
            self.data["msg_type"] == 4])
        pos = RawData.position_extraction(
            self.data.loc[self.data["msg_type"] == 3], lat_ref, lon_ref
        )
        result = pd.concat(
            [ide, vel, pos], axis=0, sort=False, ignore_index=True
        )
        result.sort_values(by=["mintime", "icao24"], inplace=True)
        return result

    def get_type(self, inplace=True):
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

    def to_beast(self) -> pd.Series:
        df_beast = self.data[["timestamp", "rawmsg"]].copy()
        ref_time = self.data["timestamp"].iloc[0]
        df_beast["time"] = self.data["timestamp"] - ref_time
        df_beast.time = df_beast.time.apply(lambda row: row.value) * 12 * 10e6
        df_beast.time = df_beast.time.apply(
            lambda row: hex(int(row))[2:].zfill(12)
        )
        df_beast["message"] = "@" + df_beast.time + df_beast.rawmsg
        return df_beast["message"]

    def to_sbs(self: pd.DataFrame) -> pd.DataFrame:
        """
        Take a pandas dataframe with flight information and convert it into a
        pandas dataframe in SBS format.

        Parameters
        ----------
        data : pd.DataFrame
            Must include following data :
                mintime : time data for each message ;
                icao24 : aircraft ID ;
                msg_type : wether pos (3), ide (1) or vel (4) ;
                callsign ;
                speed ;
                altitude ;
                vertical rate ;
                longitude ;
                latitude ;
                heading ;

        Returns
        -------
        sbs_data : pd.DataFrame
            Return a SBS like pandas dataframe. Call the pd.DataFrame.to_csv()
            function on it to get your .BST file. Don't forget to set index and
            header parameter to False.

        """

        sbs_data = self.data.copy(deep=True)

        sbs_data["msg"] = "MSG"
        sbs_data["msg_type2"] = "3"
        sbs_data["icao24"] = sbs_data["icao24"].str.upper()
        sbs_data["icao24_dec"] = sbs_data.icao24.apply(int, base=16)

        sbs_data["icao24_2"] = sbs_data.icao24_dec

        sbs_data["date"] = pd.to_datetime(sbs_data.mintime).dt.strftime(
            "%Y/%m/%d"
        )

        sbs_data["time"] = (
            pd.to_datetime(sbs_data.mintime)
            .dt.strftime("%H:%M:%S.%f")
            .str.slice(0, -3, 1)
        )

        sbs_data["date_2"] = sbs_data["date"]
        sbs_data["time_2"] = sbs_data["time"]

        alt = [
            value
            for value in ["altitude", "alt"]
            if value in list(sbs_data.columns)
        ]
        sbs_data["alt"] = sbs_data[alt]  # *3.28084

        sbs_data["alt"] = sbs_data.alt.round().astype("Int64")

        vel = [
            value
            for value in ["vel", "velocity", "speed"]
            if value in list(sbs_data.columns)
        ]
        sbs_data["velocity"] = sbs_data[vel]  # *1.94384

        lat = [
            value
            for value in ["latitude", "lat"]
            if value in list(sbs_data.columns)
        ]

        lon = [
            value
            for value in ["longitude", "lon"]
            if value in list(sbs_data.columns)
        ]

        vert = [
            value
            for value in ["vertrate", "vertical_rate"]
            if value in list(sbs_data.columns)
        ]

        sbs_data["squawk"] = np.nan
        sbs_data.loc[sbs_data["msg_type"] == 3, "alert"] = "0"
        sbs_data.loc[sbs_data["msg_type"] == 3, "emergency"] = "0"
        sbs_data.loc[sbs_data["msg_type"] == 3, "spi"] = "0"
        sbs_data.loc[sbs_data["msg_type"] == 3, "surface"] = "0"

        cols = [
            "msg",
            "msg_type",
            "msg_type2",
            "icao24_dec",
            "icao24",
            "icao24_2",
            "date",
            "time",
            "date_2",
            "time_2",
            "callsign",
            "alt",
            "velocity",
            "heading",
            lat[0],
            lon[0],
            vert[0],
            "squawk",
            "alert",
            "emergency",
            "spi",
            "surface",
        ]

        sbs_data = sbs_data[cols]
        sbs_data.sort_values(by=["time", "icao24"], inplace=True)

        return sbs_data
