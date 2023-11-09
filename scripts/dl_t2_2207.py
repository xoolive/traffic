from __future__ import annotations
from traffic.data import (  # noqa: F401
    aixm_airspaces,
    aixm_navaids,
    aixm_airways,
)
from traffic.core import Traffic, Flight, FlightPlan
import pandas as pd
from time import time

# from intervals import IntervalCollection
from pathlib import Path
import datetime

from typing import Any, Dict, cast  # noqa: F401
from stats_devs_pack.functions_heuristic import predict_fp

from traffic.core.mixins import DataFrameMixin
from function_parsing_sector import sectors_openings

extent = "LFBBBDX"
altitude_min = 20000
# metadata = cycle airac complet
metadata = pd.read_parquet("A2207_old.parquet")
metadata_simple = (
    metadata.groupby("flight_id", as_index=False)
    .last()
    .eval("icao24 = icao24.str.lower()")
)

# Directory containing the Parquet files
directory = "../../LFBB_A2207"
concatenated_df = pd.read_parquet(directory)


t = (
    Traffic(concatenated_df)
    .drop(columns=["onground"])
    .clip(aixm_airspaces[extent])
    .query(f"altitude>{altitude_min}")
    .eval(max_workers=4)
)
# t = t[0:1000]

start = time()
t2 = (
    t.iterate_lazy(iterate_kw=dict(by=metadata_simple))
    .resample("1s")
    .eval(desc="", max_workers=4)
)

# t2.data["onground"] = t2.data["onground"].astype(bool)
print(f"total time : {time()-start}")
t2.to_parquet("t2_0722_noonground.parquet")
