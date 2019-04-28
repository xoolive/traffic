from pathlib import Path
from typing import Union

from ...core import Flight, Traffic


def get_flight(filename: str, directory: Path) -> Union[Flight, Traffic]:
    flight: Union[None, Flight, Traffic] = Traffic.from_file(
        directory / f"{filename}.json.gz"
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


def assign_id(t: Union[Traffic, Flight], name: str) -> Union[Traffic, Flight]:
    if "flight_id" in t.data.columns:
        return t
    if isinstance(t, Traffic):
        return t.assign_id()
    else:
        return t.assign(flight_id=name)
