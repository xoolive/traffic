from __future__ import annotations

import json
from typing import NamedTuple
from zipfile import ZipFile, ZipInfo

import numpy as np
import pandas as pd

from ...core import Flight, Traffic, tqdm
from ...data.basic.navaid import Navaids
from .mendeley import Mendeley


class Entry(NamedTuple):
    flight: Flight
    flight_plan: pd.DataFrame
    clearances: pd.DataFrame


rename_columns = {
    "time_stamp": "timestamp",
    "I062/105.lat": "latitude",
    "I062/105.lon": "longitude",
    "I062/136.measured_flight_level": "flight_level",
    "I062/185.vx": "vx",
    "I062/185.vy": "vy",
    "I062/220.rocd": "vertical_rate",
    "I062/380.subitem3.ag_hdg": "heading",
    "I062/380.subitem7.altitude": "selected_altitude",
    "I062/380.subitem26.ias": "IAS",
    "I062/380.subitem27.mach": "Mach",
}


class SCAT:
    """This class parses a dataset of 170,000 flights.

    The Swedish Civil Air Traffic Control (SCAT) dataset contains detailed data
    of almost 170,000 flights as well as weather forecasts and airspace data
    collected from the air traffic control system in the Swedish flight
    information region. The flight data includes system updated flight plans,
    clearances from air traffic control, surveillance data and trajectory
    prediction data. The data is divided into 13 different weeks of data spread
    over one year. The data is limited to scheduled flights where for example
    military and private aircraft has been removed from the recorded data.

    https://data.mendeley.com/datasets/8yn985bwz5/

    """

    traffic: Traffic
    flight_plans: pd.DataFrame
    clearances: pd.DataFrame
    waypoints: Navaids
    weather: pd.DataFrame

    def parse_zipinfo(self, zf: ZipFile, file_info: ZipInfo) -> Entry:
        with zf.open(file_info.filename, "r") as fh:
            content_bytes = fh.read()
        decoded = json.loads(content_bytes.decode())
        flight_id = str(decoded["id"])  # noqa: F841

        flight_plan = (
            pd.json_normalize(decoded["fpl"]["fpl_plan_update"])
            .rename(columns=rename_columns)
            .eval(
                """
            timestamp = @pd.to_datetime(timestamp, utc=True, format="mixed")
            flight_id = @flight_id
            """,
                engine="python",
            )
        )

        clearance = (
            pd.json_normalize(decoded["fpl"]["fpl_clearance"])
            .rename(columns=rename_columns)
            .eval(
                """
            timestamp = @pd.to_datetime(timestamp, utc=True, format="mixed")
            flight_id = @flight_id
            """,
                engine="python",
            )
        )

        fpl_base, *_ = decoded["fpl"]["fpl_base"]
        df = (
            pd.json_normalize(decoded["plots"])
            .rename(columns=rename_columns)
            .eval(
                """
        timestamp = @pd.to_datetime(time_of_track, utc=True, format="mixed")
        altitude = 100 * flight_level
        origin = @fpl_base['adep']
        destination = @fpl_base['ades']
        typecode = @fpl_base['aircraft_type']
        callsign = @fpl_base['callsign']
        flight_id = @flight_id
        icao24 = "000000"
        """,
                engine="python",
            )
        )
        return Entry(Flight(df), flight_plan, clearance)

    def parse_waypoints(self, zf: ZipFile, file_info: ZipInfo) -> Navaids:
        rename_columns = {
            "lat": "latitude",
            "lon": "longitude",
        }
        with zf.open(file_info.filename, "r") as fh:
            content_bytes = fh.read()
        centers = json.loads(content_bytes.decode())

        fixes = []
        for center in centers:
            points = pd.json_normalize(center["points"])
            points["type"] = "FIX"
            points["altitude"] = None
            points["frequency"] = None
            points["magnetic_variation"] = None
            points["description"] = f"Center: {center['name']}"
            fixes.append(points.rename(columns=rename_columns))
        df = pd.concat(fixes).drop_duplicates(ignore_index=True)
        waypoints = Navaids(data=df)
        waypoints.priority = -1  # prefer over default navaids
        return waypoints

    def parse_weather(self, zf: ZipFile, file_info: ZipInfo) -> pd.DataFrame:
        rename_columns = {
            "alt": "altitude",
            "lat": "latitude",
            "lon": "longitude",
            "temp": "temperature",
            "time": "timestamp",
            "wind_dir": "wind_direction",
            "wind_spd": "wind_speed",
        }
        with zf.open(file_info.filename, "r") as fh:
            content_bytes = fh.read()
        decoded = json.loads(content_bytes.decode())
        return (
            pd.json_normalize(decoded)
            .rename(columns=rename_columns)
            .eval(
                """
            timestamp = @pd.to_datetime(timestamp, utc=True, format="mixed")
            """,
                engine="python",
            )
        )

    def __init__(
        self,
        ident: str,
        nflights: None | int = None,
        include_waypoints: bool = False,
        include_weather: bool = False,
    ) -> None:
        mendeley = Mendeley("8yn985bwz5")
        filename = mendeley.get_data(ident)

        clearances = []
        flights = []
        flight_plans = []

        with ZipFile(filename, "r") as zf:
            all_files = zf.infolist()
            total_flights = len(all_files) - 2
            nflights = (
                min(nflights, total_flights)
                if nflights is not None
                else total_flights
            )
            info_list = all_files[:nflights]
            if include_waypoints:
                info_list.append(all_files[-2])
            if include_weather:
                info_list.append(all_files[-1])

            for file_info in tqdm(info_list):
                if "airspace" in file_info.filename:
                    self.waypoints = self.parse_waypoints(zf, file_info)
                    continue

                if "grib_meteo" in file_info.filename:
                    self.weather = self.parse_weather(zf, file_info)
                    continue

                entry = self.parse_zipinfo(zf, file_info)
                flights.append(entry.flight)
                flight_plans.append(entry.flight_plan)
                clearances.append(entry.clearances)

        self.flight_plans = pd.concat(flight_plans)
        self.clearances = pd.concat(clearances)

        t = Traffic.from_flights(flights)
        assert t is not None
        self.traffic = t.assign(
            track=lambda df: (90 - np.angle(df.vx + 1j * df.vy, deg=True))
            % 360,
            groundspeed=lambda df: np.abs(df.vx + 1j * df.vy) / 0.514444,
        ).drop(
            columns=[
                "time_of_track",
                # "latitude",
                # "longitude",
                # "flight_level",
                # "vx",
                # "vy",
                "I062/200.adf",
                "I062/200.long",
                "I062/200.trans",
                "I062/200.vert",
                # "vertical_rate",
                "I062/380.subitem13.baro_vert_rate",
                # "IAS",
                # "Mach",
                "I062/380.subitem3.mag_hdg",
                "I062/380.subitem6.altitude",
                "I062/380.subitem6.sas",
                "I062/380.subitem6.source",
                "I062/380.subitem7.ah",
                # "selected_altitude",
                "I062/380.subitem7.am",
                "I062/380.subitem7.mv",
                # "timestamp",
                # "altitude",
                # "origin",
                # "destination",
                # "typecode",
                # "callsign",
                # "flight_id",
            ]
        )
