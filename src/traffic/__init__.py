import configparser
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import dotenv
from appdirs import user_cache_dir, user_config_dir

import pandas as pd

from . import visualize  # noqa: F401

__version__ = version("traffic")
__all__ = ["cache_path", "config_dir", "config_file", "tqdm_style"]

# Set up the library root logger
_log = logging.getLogger(__name__)

dotenv.load_dotenv()

# -- Configuration management --

if (xdg_config := os.environ.get("XDG_CONFIG_HOME")) is not None:
    config_dir = Path(xdg_config) / "traffic"
else:
    config_dir = Path(user_config_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():  # coverage: ignore
    config_template = (Path(__file__).parent / "traffic.conf").read_text()
    config_dir.mkdir(parents=True)
    config_file.write_text(config_template)

config = configparser.ConfigParser()
config.read(config_file.as_posix())


class Resolution(TypedDict, total=False):
    category: str
    name: str
    environment_variable: str
    default: str


NAME_RESOLUTION: dict[str, Resolution] = {
    # Cache configuration
    "cache_dir": dict(
        environment_variable="TRAFFIC_CACHE_PATH",
        category="cache",
        name="path",
    ),
    "cache_expiration": dict(
        environment_variable="TRAFFIC_CACHE_EXPIRATION",
        category="cache",
        name="expiration",
    ),
    "aixm_path_str": dict(
        environment_variable="TRAFFIC_AIXM_PATH",
        category="global",
        name="aixm_path",
    ),
    "nm_path_str": dict(
        environment_variable="TRAFFIC_NM_PATH",
        category="global",
        name="nm_path",
    ),
    # Should we get a tqdm progress bar
    "tqdm_style": dict(
        environment_variable="TRAFFIC_TQDM_STYLE",
        category="global",
        name="tqdm_style",
        default="auto",
    ),
    "aircraft_db": dict(
        environment_variable="TRAFFIC_AIRCRAFTDB",
        category="aircraft",
        name="database",
    ),
}


# Check the config file for a cache directory. If not present
# then use the system default cache path


def get_config(
    category: None | str = None,
    name: None | str = None,
    environment_variable: None | str = None,
    default: None | str = None,
) -> None | str:
    if category is not None and name is not None:
        if value := config.get(category, name, fallback=None):
            return value

    if environment_variable is not None:
        return os.environ.get(environment_variable)

    if default is not None:
        return default

    return None


cache_dir = get_config(**NAME_RESOLUTION["cache_dir"])
if cache_dir is None:
    cache_dir = user_cache_dir("traffic")
cache_path = Path(cache_dir)
if not cache_path.exists():
    cache_path.mkdir(parents=True)

_cache_expiration_str = get_config(**NAME_RESOLUTION["cache_expiration"])
cache_expiration = (
    pd.Timedelta(_cache_expiration_str)
    if _cache_expiration_str is not None
    else None
)
_log.info(f"Selected cache_expiration: {cache_expiration}")

aircraft_db_path = get_config(**NAME_RESOLUTION["aircraft_db"])
_log.info(f"Selected aircraft_db path: {aircraft_db_path}")
aixm_path_str = get_config(**NAME_RESOLUTION["aixm_path_str"])
_log.info(f"Selected aixm path: {aixm_path_str}")
nm_path_str = get_config(**NAME_RESOLUTION["nm_path_str"])
_log.info(f"Selected nm path: {nm_path_str}")
tqdm_style = get_config(**NAME_RESOLUTION["tqdm_style"])
_log.info(f"Selected tqdm style: {tqdm_style}")
