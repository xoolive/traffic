import sys
from functools import lru_cache
from pathlib import Path
from typing import Union, cast

import pandas as pd

from traffic.core import Airspace

from ...core import Flight, Traffic

_current_dir = Path(__file__).parent
__all__ = list(f.stem[:-5] for f in _current_dir.glob("**/*.json.gz"))


@lru_cache()
def get_flight(filename: str, directory: Path) -> Union[Flight, Traffic]:
    flight: Union[None, Flight, Traffic] = Traffic.from_file(
        directory / f"{filename}.json.gz", dtype={"icao24": str}
    )
    if flight is None:
        raise RuntimeError(f"File {filename}.json.gz not found in {directory}")
    icao24 = set(flight.data.icao24)
    if len(icao24) == 1:
        if Flight(flight.data).split("1H").sum() == 1:
            # easier way to cast...
            flight = Flight(flight.data)
    # -- Dealing with time-like features --
    if "hour" in flight.data.columns:
        flight = flight.assign(
            hour=lambda df: pd.to_datetime(df.hour * 1e9).dt.tz_localize("utc")
        )
    if "last_position" in flight.data.columns:
        flight = flight.assign(
            last_position=lambda df: pd.to_datetime(
                df.last_position * 1e6
            ).dt.tz_localize("utc")
        )
    return flight.assign(
        timestamp=lambda df: df.timestamp.dt.tz_localize("utc")
    )


def get_sample(module, name: str):
    if sys.version_info >= (3, 7):
        return getattr(module, name)
    path = Path(module.__file__).parent
    return get_flight(name, path)


def assign_id(t: Union[Traffic, Flight], name: str) -> Union[Traffic, Flight]:
    if "flight_id" in t.data.columns:
        return t
    if isinstance(t, Traffic):
        return cast(Traffic, t.assign_id().eval())
    else:
        return t.assign(flight_id=name)


@lru_cache()
def __getattr__(name: str) -> Union[Flight, Traffic]:
    filelist = list(_current_dir.glob(f"**/{name}.json.gz"))
    if len(filelist) == 0:
        msg = f"File {name}.json.gz not found in available samples"
        raise AttributeError(msg)
    return get_flight(name, filelist[0].parent)


airbus_tree: Flight = cast(Flight, __getattr__("airbus_tree"))
belevingsvlucht: Flight = cast(Flight, __getattr__("belevingsvlucht"))
elal747: Flight = cast(Flight, __getattr__("elal747"))
texas_longhorn: Flight = cast(Flight, __getattr__("texas_longhorn"))
quickstart: Traffic = cast(Traffic, __getattr__("quickstart"))
switzerland: Traffic = cast(Traffic, __getattr__("switzerland"))

lfbo_tma = Airspace.from_file(
    Path(__file__).parent / "airspaces" / "LFBOTMA.json"
)
