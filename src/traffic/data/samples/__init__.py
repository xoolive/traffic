from __future__ import annotations

import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Union, cast

import pandas as pd

from ...core import Airspace, Flight, Traffic

_current_dir = Path(__file__).parent
__all__ = [
    *sorted(f.stem[:-5] for f in _current_dir.glob("**/*.json.gz")),
    "sample_dump1090",
]


@lru_cache()
def get_flight(filename: str, directory: Path) -> Flight | Traffic:
    flight: Union[None, Flight, Traffic] = None
    if (fh := directory / f"{filename}.json.gz").exists():
        flight = Traffic.from_file(fh, dtype={"icao24": str})
    if (fh := directory / f"{filename}.jsonl").exists():
        flight = Traffic.from_file(fh, dtype={"icao24": str})
    if flight is None:
        raise RuntimeError(
            f"File {filename}.json[l,.gz] not found in {directory}"
        )
    icao24 = set(flight.data.icao24) if "icao24" in flight.data.columns else []
    if len(icao24) <= 1:
        if Flight(flight.data).split("1h").sum() == 1:
            # easier way to cast...
            flight = Flight(flight.data)
    # -- Dealing with time-like features --
    if "hour" in flight.data.columns:
        flight = flight.assign(
            hour=lambda df: pd.to_datetime(df.hour * 1e9).dt.tz_localize("utc")
        )
    if "last_position" in flight.data.columns:
        last_pos = pd.to_datetime(flight.data.last_position.to_numpy() * 1e6)
        if last_pos.tz is None:
            last_pos = last_pos.tz_localize("utc")
        flight = flight.assign(last_position=last_pos)
    return flight


def get_sample(
    module: types.ModuleType, name: str
) -> Union[None, "Flight", "Traffic"]:
    return getattr(module, name)  # type: ignore


def assign_id(t: Union[Traffic, Flight], name: str) -> Union[Traffic, Flight]:
    if "flight_id" in t.data.columns:
        return t
    if isinstance(t, Traffic):
        return cast(Traffic, t.assign_id().eval())
    else:
        return t.assign(flight_id=name)


airbus_tree: Flight
belevingsvlucht: Flight
elal747: Flight
full_flight_short: Flight
lfbo_tma: Airspace
noisy: Flight
quickstart: Traffic
sample_dump1090: Path
switzerland: Traffic
texas_longhorn: Flight
zurich_airport: Traffic


@lru_cache()
def __getattr__(name: str) -> Any:
    if name == "sample_dump1090":
        return Path(_current_dir / "dump1090" / "sample_dump1090.bin")
    if name == "lfbo_tma":
        return Airspace.from_file(_current_dir / "airspaces" / "LFBOTMA.json")
    filelist = list(_current_dir.glob(f"**/{name}.json*"))
    if len(filelist) == 0:
        msg = f"File {name}.json.gz not found in available samples"
        raise AttributeError(msg)
    return get_flight(name, filelist[0].parent)
