from __future__ import annotations
from traffic.data import (  # noqa: F401
    aixm_airspaces,
    aixm_navaids,
    aixm_airways,
)
from traffic.core import Traffic
import pandas as pd

# here, extent is the Bordeaux centre, altitude_min is FL200
extent = "LFBBBDX"
altitude_min = 20000

# for each flight, metadata contains icao24, callsign, flight id and flight plan
metadata = pd.read_parquet("A2207_old.parquet")
metadata_simple = (
    metadata.groupby("flight_id", as_index=False)
    .last()
    .eval("icao24 = icao24.str.lower()")
)

# directory contains the trajectories as parquet files
directory = "../../LFBB_A2207"
concatenated_df = pd.read_parquet(directory)

# first filter on the data
# we originally the "onground" value, but we drop it here as it is not useful
t = (
    Traffic(concatenated_df)
    .drop(columns=["onground"])
    .clip(aixm_airspaces[extent])
    .query(f"altitude>{altitude_min}")
    .eval(max_workers=4)
)

# in t2, we assign flight ids using metadata_simple
t2 = (
    t.iterate_lazy(iterate_kw=dict(by=metadata_simple))
    .resample("1s")
    .eval(desc="", max_workers=4)
)

# filter and resample
assert t2 is not None
t2 = t2.filter().resample("1s").eval(max_workers=4)  # pour retest
assert t2 is not None

t2.to_parquet("test_format_data_1.parquet")
