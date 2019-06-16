import sys
from functools import lru_cache
from pathlib import Path
from typing import Union

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
        # easier way to cast...
        flight = Flight(flight.data)
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
        return t.assign_id()
    else:
        return t.assign(flight_id=name)


@lru_cache()
def __getattr__(name: str) -> Union[Flight, Traffic]:
    filelist = list(_current_dir.glob(f"**/{name}.json.gz"))
    if len(filelist) == 0:
        msg = f"File {name}.json.gz not found in available samples"
        raise AttributeError(msg)
    return get_flight(name, filelist[0].parent)
