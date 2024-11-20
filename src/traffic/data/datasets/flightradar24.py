from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, overload

import numpy as np
import pandas as pd

from ...core import Flight, Traffic, tqdm

root_folder = Path(__file__).parent.parent.parent.parent.parent
fr24_folder = root_folder / "tests" / "fr24"


class FlightRadar24:
    @classmethod
    def from_csv(cls, filename: str | Path) -> Flight:
        """Parses data as downloaded from FlightRadar24 website history page.

        - Data as downloaded from the history webpage:

          >>> Flight.from_fr24(fr24_folder / "JL516_3376ab31.csv")
          Flight(icao24=None, callsign='JAL516')
          >>> _.duration
          Timedelta('0 days 01:24:28')

        - Data as downloaded from a blog post, when they name it "granular"

          >>> f = fr24_folder / "JL516_Flightradar24_ADS-B_data_granular.csv"
          >>> Flight.from_fr24(f)
          Flight(icao24='8467d8', callsign='JAL516')
          >>> _.duration
          Timedelta('0 days 01:07:12.125000')
        """
        data = pd.read_csv(filename)
        if "UTC" in data.columns:  # That's the web version
            data = (
                data.rename(
                    columns=dict(
                        Timestamp="timestamp",
                        Callsign="callsign",
                        Altitude="altitude",
                        Speed="groundspeed",
                        Direction="track",
                    )
                )
                .eval(
                    """
                timestamp = @pd.to_datetime(timestamp, unit='s', utc=True)
                latitude = Position.str.split(",").str[0].astype("float")
                longitude = Position.str.split(",").str[1].astype("float")
                """
                )
                .drop(columns=["UTC", "Position"])
            )
        elif "network_time" in data.columns:  # That's the granular version
            data = (
                data.rename(
                    columns=dict(
                        network_time="timestamp",
                        hex="icao24",
                        speed="groundspeed",
                        vspeed="vertical_speed",
                    )
                )
                .eval(
                    """
                timestamp = timestamp.str.replace("Z", "")
                timestamp = @pd.to_datetime(timestamp, utc=True)
                icao24 = icao24.str.slice(2)
                """
                )
                .drop(
                    columns=[
                        col for col in data.columns if col.startswith("Unnamed")
                    ]
                )
            )
        else:
            raise ValueError("Couldn't detect the proper CSV format")

        return Flight(data.assign(callsign=data.callsign.fillna("").max()))

    @classmethod
    def from_json(cls, filename: str | Path) -> Flight:
        """Parses data as downloaded by FlightRadar24 website.

        >>> from traffic.data.samples import fr24
        >>> folder = Path(fr24.__file__).parent
        >>> Flight.from_fr24(folder / "3376ab31.json")
        Flight('3376ab31', icao24='8467d8', callsign='JAL516')
        >>> _.duration
        Timedelta('0 days 01:24:28')

        >>> Flight.from_fr24(folder / "2ce4f83f.json")
        Flight('2ce4f83f', icao24='ae503d', callsign='SPAR19')
        >>> _.Mach_max
        .8

        """
        filename = Path(filename)
        json_data = json.loads(filename.read_text())
        flight = json_data["result"]["response"]["data"]["flight"]
        data = pd.json_normalize(flight["track"])
        data = data.rename(columns=dict(heading="track"))
        if flight["availability"]["ems"]:
            ems_data = (
                pd.json_normalize(
                    [elt["ems"] for elt in flight["track"] if elt["ems"]]
                )
                .eval("mach = mach / 1000")
                .rename(columns=dict(ts="timestamp"))
            )
            data = pd.concat([data, ems_data]).sort_values("timestamp")
        data = (
            data.eval(
                "timestamp = @pd.to_datetime(timestamp, unit='s', utc=True)"
            )
            .assign(
                flight_id=flight["identification"]["id"],
                callsign=flight["identification"]["callsign"],
                origin=flight["airport"]["origin"]["code"]["icao"],
                destination=flight["airport"]["destination"]["code"]["icao"],
                icao24=flight["aircraft"]["identification"]["modes"].lower(),
                typecode=flight["aircraft"]["model"]["code"],
            )
            .rename(
                columns={
                    "altitude.feet": "altitude",
                    "speed.kts": "groundspeed",
                    "verticalSpeed.fpm": "vertical_speed",
                    "ias": "IAS",
                    "tas": "TAS",
                    "mach": "Mach",
                    "trueTrack": "track",
                    "rollAngle": "roll",
                    "altGPS": "geoaltitude",
                }
            )
            .drop(
                columns=[
                    "altitude.meters",
                    "speed.kmh",
                    "speed.mph",
                    "verticalSpeed.ms",
                ]
            )
            .dropna(axis=1, how="all")
        )
        return Flight(data)

    @classmethod
    def from_archive(
        cls,
        metadata: str | Path,
        trajectories: str | Path,
        **kwargs: Any,
    ) -> Traffic:  # coverage: ignore
        """Parses data as usually provided by FlightRadar24.

        When FlightRadar24 provides data excerpts from their database, they
        usually provide:

        :param metadata: a CSV file with metadata

        :param trajectories: a zip file containing one file per flight with
            trajectory information.

        :return: a regular Traffic object.
        """
        fr24_meta = pd.read_csv(metadata)

        def extract_flights(filename: str | Path) -> Iterator[pd.DataFrame]:
            with zipfile.ZipFile(filename) as zfh:
                for fileinfo in tqdm(zfh.infolist()):
                    with zfh.open(fileinfo) as fh:
                        stem = fileinfo.filename.split(".")[0]
                        flight_id = stem.split("_")[1]
                        b = BytesIO(fh.read())
                        b.seek(0)
                        yield pd.read_csv(b).assign(flight_id=(flight_id))

        df = pd.concat(extract_flights(trajectories))

        return Traffic(
            df.rename(columns=dict(heading="track"))
            .merge(
                fr24_meta.rename(
                    columns=dict(
                        equip="typecode",
                        schd_from="origin",
                        schd_to="destination",
                    )
                ).assign(
                    flight_id=fr24_meta.flight_id.astype(str),
                    icao24=fr24_meta.aircraft_id.apply(hex)
                    .str[2:]
                    .str.pad(6, "left", fillchar="0"),
                    callsign=fr24_meta.callsign.fillna(""),
                    diverted=np.where(
                        fr24_meta.schd_to == fr24_meta.real_to,
                        np.nan,  # None would cause a typing error \o/
                        fr24_meta.real_to,
                    ),
                ),
                on="flight_id",
            )
            .eval(
                "timestamp = @pd.to_datetime(snapshot_id, utc=True, unit='s')"
            )
        )

    @overload
    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        trajectories: None = None,
        **kwargs: Any,
    ) -> Flight: ...

    @overload
    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        trajectories: str | Path,
        **kwargs: Any,
    ) -> Traffic: ...

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        trajectories: None | str | Path = None,
        **kwargs: Any,
    ) -> Traffic | Flight:
        filename = Path(filename)
        if trajectories is None:
            if filename.suffix == ".csv":
                return cls.from_csv(filename)
            if filename.suffix == ".json":
                return cls.from_json(filename)
            raise ValueError("Unknown file type")
        return cls.from_archive(filename, trajectories)
