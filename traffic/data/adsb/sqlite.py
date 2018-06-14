import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from traffic.core import Traffic

""" TODO This parsing method is kept in a separate file as a stub for 'plugin'
for extending traffic structure.  """


def parse_sqlite(filename: Union[str, Path], with_ehs=False) -> pd.DataFrame:

    if isinstance(filename, Path):
        filename = filename.as_posix()

    connector = sqlite3.connect(filename)
    tables = ["surface", "positions", "vectors"]

    if with_ehs:
        tables += ["speed", "tracks", "meteo", "intention"]

    default_rename = {
        "lat": "latitude",
        "lon": "longitude",
        "alt": "altitude",
        "icao": "icao24",
        "speed": "ground_speed",
        "vertical": "vertical_rate",
    }

    renames: Dict[str, Dict[str, str]] = defaultdict(dict)
    renames["position"] = {"speed": "ground_speed"}
    renames["tracks"] = {"t_airspeed": "true_airspeed"}
    renames["vectors"] = {"heading": "track"}

    def posttreat(out: pd.DataFrame, more_rename=dict()) -> pd.DataFrame:
        df = out.rename(columns={**default_rename, **more_rename}).assign(
            timestamp=out.timestamp.apply(datetime.fromtimestamp),
            icao24=out.icao.apply(lambda x: hex(x)[2:]),
        )
        return df[df.callsign != "UNKNOWN"]

    return pd.concat(
        [
            posttreat(
                pd.read_sql_query(
                    f"select * from {table} order by timestamp", connector
                ),
                renames[table],
            )
            for table in tables
        ]
    ).sort_values("timestamp")


Traffic._parse_extension[".db"] = parse_sqlite
