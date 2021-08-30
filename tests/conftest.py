import os
import shutil
from pathlib import Path

from traffic.data import opensky


def pytest_configure(config):

    # Some of the tests rely on data which is no longer available on Opensky
    # Impala shell at the time
    cache_dir = Path(__file__).parent.parent / "data" / "opensky_cache"
    for p in cache_dir.glob("*"):
        new_path = opensky.cache_dir / p.name
        shutil.copy(str(p), str(new_path))  # str necessary for Python <= 3.7

    # When new tests are implemented, we should be able to run them on CI
    username = os.environ.get("OPENSKY_USERNAME", None)
    password = os.environ.get("OPENSKY_PASSWORD", None)
    if username is not None and password is not None:
        opensky.username = username
        opensky.password = password
        opensky.auth = (username, password)
