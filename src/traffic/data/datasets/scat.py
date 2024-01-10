from __future__ import annotations

import json
from typing import NamedTuple
from zipfile import ZipFile, ZipInfo

import numpy as np
import pandas as pd

from ...core import Flight, Traffic, tqdm
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
                """
                )
            )

            clearance = (
                pd.json_normalize(decoded["fpl"]["fpl_clearance"])
                .rename(columns=rename_columns)
                .eval(
                    """
                timestamp = @pd.to_datetime(timestamp, utc=True, format="mixed")
                flight_id = @flight_id
                """
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
            """
                )
            )
            return Entry(Flight(df), flight_plan, clearance)

    def __init__(self, ident: str, nflights: None | int = None) -> None:
        mendeley = Mendeley("8yn985bwz5")
        filename = mendeley.get_data(ident)

        clearances = []
        flights = []
        flight_plans = []

        with ZipFile(filename, "r") as zf:
            info_list = zf.infolist()
            if nflights is not None:
                info_list = info_list[:nflights]
            for file_info in tqdm(info_list):
                if "airspace" in file_info.filename:
                    continue

                if "grib_meteo" in file_info.filename:
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
