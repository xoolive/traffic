import logging
from pathlib import Path
from typing import List, Optional, Union, cast

import numpy as np
import pandas as pd

from traffic.core import Traffic
from traffic.core.aero import vtas2cas
from traffic.core.time import timelike
from traffic.data import aircraft


def fmt_timedelta(x: pd.Timestamp) -> str:
    return (
        f"{24 * x.components.days + x.components.hours}:"
        f"{x.components.minutes}:{x.components.seconds}"
    )


def to_bluesky(
    traffic: Traffic,
    filename: Union[str, Path],
    minimum_time: Optional[timelike] = None,
) -> None:
    """Generates a Bluesky scenario file."""

    if minimum_time is not None:
        traffic = traffic.after(minimum_time)

    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    altitude = (
        "baro_altitude"
        if "baro_altitude" in traffic.data.columns
        else "altitude"
    )

    if "typecode" not in traffic.data.columns:
        traffic = Traffic(
            traffic.data.merge(
                aircraft.data[["icao24", "typecode"]].drop_duplicates("icao24"),
                on="icao24",
                how="inner",
            )
        )

    if "cas" not in traffic.data.columns:
        traffic = Traffic(
            traffic.data.assign(
                cas=vtas2cas(traffic.data.ground_speed, traffic.data[altitude])
            )
        )

    with filename.open("w") as fh:
        t_delta = traffic.data.timestamp - traffic.start_time
        data = (
            traffic.assign_id()
            .data.groupby("flight_id")
            .filter(lambda x: x.shape[0] > 3)
            .assign(timedelta=t_delta.apply(fmt_timedelta))
            .sort_values(by="timestamp")
        )

        for column in data.columns:
            data[column] = data[column].astype(np.str)

        is_created: List[str] = []
        is_deleted: List[str] = []

        start_time = cast(pd.Timestamp, traffic.start_time).time()
        fh.write(f"00:00:00> TIME {start_time}\n")

        # Add some bluesky command for the visualisation
        # fh.write("00:00:00>trail on\n")
        # fh.write("00:00:00>ssd conflicts\n")

        # We remove an object when it's its last data point
        buff = data.groupby("flight_id").timestamp.max()
        dd = pd.DataFrame(
            columns=["timestamp"], data=buff.values, index=buff.index.values
        )
        map_icao24_last_point = {}
        for i, v in dd.iterrows():
            map_icao24_last_point[i] = v[0]

        # Main loop to write lines in the scenario file
        for _, v in data.iterrows():
            if v.flight_id not in is_created:
                # If the object is not created then create it
                is_created.append(v.flight_id)
                fh.write(
                    f"{v.timedelta}> CRE {v.callsign} {v.typecode} "
                    f"{v.latitude} {v.longitude} {v.track} "
                    f"{v[altitude]} {v.cas}\n"
                )

            elif v.timestamp == map_icao24_last_point[v.flight_id]:
                # Remove an aircraft when no data are available
                if v.flight_id not in is_deleted:
                    is_deleted.append(v.flight_id)
                    fh.write(f"{v.timedelta}> DEL {v.callsign}\n")

            elif v.flight_id not in is_deleted:
                # Otherwise update the object position
                fh.write(
                    f"{v.timedelta}> MOVE {v.callsign} "
                    f"{v.latitude} {v.longitude} {v[altitude]} "
                    f"{v.track} {v.cas} {v.vertical_rate}\n"
                )

        logging.info(f"Scenario file {filename} written")


def _onload():
    setattr(Traffic, "to_bluesky", to_bluesky)
