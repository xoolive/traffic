from pathlib import Path

from ...core import Traffic

calibration = Traffic.from_file(Path(__file__).parent / "calibration.pkl.gz")
