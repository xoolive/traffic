from pathlib import Path

from ...core import Flight


def get_flight(filename: str, directory: Path) -> Flight:
    flight = Flight.from_file(directory / f"{filename}.json.gz")
    if flight is None:
        raise RuntimeError(f"File {filename}.json.gz not found in {directory}")
    return flight.assign(
        timestamp=lambda df: df.timestamp.dt.tz_localize("utc")
    )
