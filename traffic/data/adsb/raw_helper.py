from pathlib import Path

import pandas as pd
import numpy as np
from pyModeS import adsb
from pyModeS.decoder.bds.bds08 import callsign
from requests import Session

from .opensky_impala import Impala


class RawHelper(Impala):

    _raw_columns = [
        'mintime',
        'maxtime',
        'msgcount',
        'rawmsg',
        'icao24',
        'msg_type'
    ]

    def __init__(self,
                 username: str,
                 password: str,
                 cache_dir: Path,
                 session: Session,
                 proxy_command: str,) -> None:
        super().__init__(username, password, cache_dir, proxy_command)
        self.session = session

    def get_raw(self, *args, **kwargs):
        """Get EHS message from the OpenSky Impala shell.

        You may pass requests based on time ranges, callsigns, aircraft, areas,
        serial numbers for receivers, or airports of departure or arrival.

        The method builds appropriate SQL requests, caches results and formats
        data into a proper pandas DataFrame. Requests are split by hour (by
        default) in case the connection fails.

        Args:
            - **start**: a string (default to UTC), epoch or datetime (native
              Python or pandas)
            - **stop** (optional): a string (default to UTC), epoch or datetime
              (native Python or pandas), *by default, one day after start*
            - **date_delta** (optional): a timedelta representing how to split
              the requests, *by default: per hour*

            More arguments to filter resulting data:

            - **callsign** (optional): a string or a list of strings (wildcards
              accepted, _ for any character, % for any sequence of characters);
            - **icao24** (optional): a string or a list of strings identifying
              the transponder code of the aircraft;
            - **serials** (optional): an integer or a list of integers
              identifying the sensors receiving the data;
            - **bounds** (optional), sets a geographical footprint. Either
              an **airspace or shapely shape** (requires the bounds attribute);
              or a **tuple of float** (west, south, east, north);

            **Airports**

            The following options build more complicated requests by merging
            information from two tables in the Impala database, resp.
            `rollcall_replies_data4` and `flights_data4`.

            - **departure_airport** (optional): a string for the ICAO
              identifier of the airport. Selects flights departing from the
              airport between the two timestamps;
            - **arrival_airport** (optional): a string for the ICAO identifier
              of the airport. Selects flights arriving at the airport between
              the two timestamps;
            - **airport** (optional): a string for the ICAO identifier of the
              airport. Selects flights departing from or arriving at the
              airport between the two timestamps;

            .. warning::

                - If both departure_airport and arrival_airport are set,
                  requested timestamps match the arrival time;
                - If airport is set, departure_airport and
                  arrival_airport cannot be specified (a RuntimeException is
                  raised).
                - It is not possible at the moment to filter both on airports
                  and on geographical bounds (help welcome!).

            **Useful options for debug**

            - **cached** (boolean, default: True): switch to False to force a
              new request to the database regardless of the cached files;
              delete previous cache files;
            - **limit** (optional, int): maximum number of records requested
              LIMIT keyword in SQL.

        """

        if kwargs.get('table_name') in self._raw_tables:
            self.raw_history(*args, **kwargs)
        else:
            pos_data = self.raw_history(table_name=self._raw_tables[0],
                                        *args, **kwargs)
            if not pos_data.empty:
                pos_data['msg_type'] = 3

            vel_data = self.raw_history(table_name=self._raw_tables[2],
                                        *args, **kwargs)
            if not vel_data.empty:
                vel_data['msg_type'] = 4

            ide_data = self.raw_history(table_name=self._raw_tables[1],
                                        *args, **kwargs)
            if not ide_data.empty:
                ide_data['msg_type'] = 1

            result = (pd.concat([pos_data,
                                 vel_data,
                                 ide_data])[self._raw_columns]
                      .sort_values("mintime"))

        try:
            return result
        except ValueError:
            print("No aircraft found.")

    @staticmethod
    def beast_converter(data: pd.DataFrame) -> pd.Series:
        df_beast = data[['timestamp', 'rawmsg']].copy()
        ref_time = data['timestamp'].iloc[0]
        df_beast['time'] = data['timestamp'] - ref_time
        df_beast.time = df_beast.time.apply(lambda row: row.value) * 12 * 10e6
        df_beast.time = df_beast.time.apply(lambda row:
                                            hex(int(row))[2:].zfill(12))
        df_beast['message'] = '@' + df_beast.time + df_beast.rawmsg
        return df_beast['message']

    @staticmethod
    def raw_cleaner(data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def position_extraction(os_data,
                            lat_ref=None, lon_ref=None) -> pd.DataFrame:

        # pandas option to get rid of SettingWithCopyWarning
        pd.options.mode.chained_assignment = None

        os_pos = os_data.loc[os_data['msg_type'] == 3]
        os_pos["oe_flag"] = os_pos["rawmsg"].apply(adsb.oe_flag)
        os_pos["altitude"] = os_pos["rawmsg"].apply(adsb.altitude)
        os_pos.sort_values(by=['icao24', 'mintime'], inplace=True)
        os_pos["inv"] = np.nan
        os_pos["inv_time"] = np.nan

        # Loop on the last 10 messages to find a message of the other parity.
        for j in range(10, 0, -1):

            # inv is the other parity message found. nan if none found.
            os_pos["inv"] = np.where(
                (os_pos.shift(j)["icao24"] == os_pos["icao24"]) &
                (os_pos["oe_flag"].shift(j) != os_pos["oe_flag"]) &
                (os_pos["mintime"].shift(j) > os_pos["mintime"] -
                 10), os_pos.shift(j)["rawmsg"], os_pos.inv)

            # inv_time is the ts associated to the message. nan if none found.
            os_pos["inv_time"] = np.where(
                (os_pos.shift(j)["icao24"] == os_pos["icao24"]) &
                (os_pos["oe_flag"].shift(j) != os_pos["oe_flag"]) &
                (os_pos["mintime"].shift(j) > os_pos["mintime"] -
                 10), os_pos.shift(j)["mintime"], os_pos.inv_time)

        if os_pos['mintime'].dtypes == np.float64:
            os_pos['mintime'] = pd.to_datetime(os_pos['mintime'], unit='s')

        if os_pos['inv_time'].dtypes == np.float64:
            os_pos['inv_time'] = pd.to_datetime(os_pos['inv_time'], unit='s')

        # apply adsb.position. TODO : calc with matrix. Ask Junzi.
        pos_tmp = os_pos.loc[~os_pos["inv"].isnull(), :].apply(
            lambda row: adsb.position(row.rawmsg,
                                      row.inv,
                                      row.mintime,
                                      row.inv_time),
            axis=1)

        pos = pd.DataFrame(pos_tmp.tolist(),
                           columns=["latitude", "longitude"],
                           index=pos_tmp.index)
        os_pos.drop(["inv", "inv_time", "oe_flag"], axis=1, inplace=True)
        result = pd.concat([os_pos, pos], axis=1, join="inner")
        return result

    @staticmethod
    def velocity_extraction(os_data) -> pd.DataFrame:
        os_vel = os_data.loc[os_data['msg_type'] == 4]
        if os_vel['mintime'].dtypes == np.float64:
            os_vel['mintime'] = pd.to_datetime(os_vel['mintime'], unit='s')
        os_vel["speed_temp"] = os_vel["rawmsg"].apply(adsb.velocity)
        os_vel[["speed", "heading", "vertical_rate", "type"]] = pd.DataFrame(
            os_vel['speed_temp'].tolist(), index=os_vel.index)
        os_vel.drop(['speed_temp', 'type'], axis=1, inplace=True)
        return os_vel

    @staticmethod
    def identification_extraction(os_data) -> pd.DataFrame:
        os_ide = os_data.loc[os_data['msg_type'] == 1]
        if os_ide['mintime'].dtypes == np.float64:
            os_ide['mintime'] = pd.to_datetime(os_ide['mintime'], unit='s')
        os_ide["callsign"] = os_ide["rawmsg"].apply(callsign)
        return os_ide

    @staticmethod
    def feature_extraction(os_data,
                           lat_ref=None, lon_ref=None) -> pd.DataFrame:
        ide = RawHelper.identification_extraction(os_data)
        vel = RawHelper.velocity_extraction(os_data)
        pos = RawHelper.position_extraction(os_data, lat_ref, lon_ref)
        result = pd.concat([ide, vel, pos],
                           axis=0,
                           sort=False,
                           ignore_index=True)
        result.sort_values(by=['mintime', 'icao24'], inplace=True)
        return result

    @staticmethod
    def sbs_converter(data: pd.DataFrame) -> pd.DataFrame:
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

        sbs_data = data.copy(deep=True)

        sbs_data['msg'] = 'MSG'
        sbs_data['msg_type2'] = '3'
        sbs_data["icao24"] = sbs_data["icao24"].str.upper()
        sbs_data['icao24_dec'] = sbs_data.icao24.apply(int, base=16)

        sbs_data['icao24_2'] = sbs_data.icao24_dec

        sbs_data['date'] = pd.to_datetime(
            sbs_data.mintime).dt.strftime('%Y/%m/%d')

        sbs_data['time'] = pd.to_datetime(
            sbs_data.mintime).dt.strftime('%H:%M:%S.%f').str.slice(0, -3, 1)

        sbs_data['date_2'] = sbs_data['date']
        sbs_data['time_2'] = sbs_data['time']

        alt = [value for value in ['altitude', 'alt']
               if value in list(sbs_data.columns)]
        sbs_data['alt'] = sbs_data[alt]  # *3.28084

        sbs_data['alt'] = sbs_data.alt.round().astype("Int64")

        vel = [value for value in ['vel', 'velocity', 'speed']
               if value in list(sbs_data.columns)]
        sbs_data['velocity'] = sbs_data[vel]  # *1.94384

        lat = [value for value in ['latitude', 'lat']
               if value in list(sbs_data.columns)]

        lon = [value for value in ['longitude', 'lon']
               if value in list(sbs_data.columns)]

        vert = [value for value in ['vertrate', 'vertical_rate']
                if value in list(sbs_data.columns)]

        sbs_data['squawk'] = np.nan
        sbs_data.loc[sbs_data['msg_type'] == 3, 'alert'] = '0'
        sbs_data.loc[sbs_data['msg_type'] == 3, 'emergency'] = '0'
        sbs_data.loc[sbs_data['msg_type'] == 3, 'spi'] = '0'
        sbs_data.loc[sbs_data['msg_type'] == 3, 'surface'] = '0'

        cols = ['msg', 'msg_type', 'msg_type2', 'icao24_dec', 'icao24',
                'icao24_2', 'date', 'time', 'date_2', 'time_2', 'callsign',
                'alt', 'velocity', 'heading', lat[0], lon[0], vert[0],
                'squawk', 'alert', 'emergency', 'spi', 'surface']

        sbs_data = sbs_data[cols]
        sbs_data.sort_values(by=['time', 'icao24'], inplace=True)

        return sbs_data
