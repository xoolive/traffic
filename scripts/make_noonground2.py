from __future__ import annotations
from traffic.data import (  # noqa: F401
    aixm_airspaces,
    aixm_navaids,
    aixm_airways,
)
from traffic.core import Traffic, Flight, FlightPlan
import pandas as pd

# from intervals import IntervalCollection
from pathlib import Path
import datetime

from typing import Any, Dict, cast  # noqa: F401
from stats_devs_pack.functions_heuristic import predict_fp
import os

from traffic.core.mixins import DataFrameMixin
from function_parsing_sector import sectors_openings

extent = "LFBBBDX"
prefix_sector = "LFBB"
file_sector = Path("../../sectors_LFBB/2022-07-BORD/2022-07-14_BORD")
margin_fl = 50  # margin for flight level
altitude_min = 20000
sector_openings = sectors_openings()
angle_precision = 2
forward_time = 20
min_distance = 200


t2 = Traffic.from_file("t2_0722_noonground.parquet")
# t2 = t2[:1000]
# assert t2 is not None
# t2 = t2.query(f"altitude>{altitude_min}")
assert t2 is not None
# t2 = t2.clip(aixm_airspaces[extent]).filter().resample("1s").eval(max_workers=4)
t2 = t2.filter().resample("1s").eval(max_workers=4)  # pour retest
assert t2 is not None
t2.to_parquet("t2_0722_noonground2.parquet")
