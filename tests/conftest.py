import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv


def pytest_configure(config: Any) -> None:
    _log = logging.getLogger()
    _log.setLevel(logging.INFO)

    load_dotenv(Path(__file__).parent / "tests.env")


@pytest.fixture(scope="session")
def requires_network() -> None:
    """Skip tests requiring external services when offline."""
    try:
        response = httpx.get(
            "https://www.flightradar24.com/",
            headers={
                "user-agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "accept": "text/html,application/xhtml+xml",
            },
            timeout=5.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"External network unavailable: {exc}")
