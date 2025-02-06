import logging
from pathlib import Path
from typing import Any

import dotenv


def pytest_configure(config: Any) -> None:
    _log = logging.getLogger()
    _log.setLevel(logging.INFO)

    dotenv.load_dotenv(Path(__file__).parent / "tests.env")
