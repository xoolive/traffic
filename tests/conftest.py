import shutil
from pathlib import Path
from typing import Any

from traffic.data import opensky


def pytest_configure(config: Any) -> None:

    # Some of the tests rely on data which is no longer available on Opensky
    # Impala shell at the time
    cache_dir = Path(__file__).parent.parent / "data" / "opensky_cache"
    for p in cache_dir.glob("*"):
        new_path = opensky.cache_dir / p.name
        shutil.copy(str(p), str(new_path))  # str necessary for Python <= 3.7
