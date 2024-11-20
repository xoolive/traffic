from pathlib import Path

from ....core import Flight

root = Path(__file__).parent


def flight(name: str) -> Flight:
    file_path = root / f"{name}.json"
    if not file_path.exists():
        raise ImportError(f"Flight {name} not found")
    return Flight.from_fr24(file_path)
