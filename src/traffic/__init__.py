import configparser
import logging
import os
from importlib.metadata import EntryPoint, entry_points, version
from pathlib import Path
from typing import Iterable

from appdirs import user_cache_dir, user_config_dir

import pandas as pd

from . import visualize  # noqa: F401

__version__ = version("traffic")
__all__ = ["config_dir", "config_file", "cache_dir", "tqdm_style"]

# Set up the library root logger
_log = logging.getLogger(__name__)

# -- Configuration management --

config_dir = Path(user_config_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():  # coverage: ignore
    config_template = (Path(__file__).parent / "traffic.conf").read_text()
    config_dir.mkdir(parents=True)
    config_file.write_text(config_template)

config = configparser.ConfigParser()
config.read(config_file.as_posix())

# Check the config file for a cache directory. If not present
# then use the system default cache path

cache_dir = Path(user_cache_dir("traffic"))

cache_dir_cfg = config.get("cache", "path", fallback="").strip()
if cache_dir_cfg != "":  # coverage: ignore
    cache_dir = Path(cache_dir_cfg)

cache_expiration_cfg = config.get("cache", "expiration", fallback="180 days")
cache_expiration = pd.Timedelta(cache_expiration_cfg)

cache_purge_cfg = config.get("cache", "purge", fallback="")
cache_no_expire = bool(os.environ.get("TRAFFIC_CACHE_NO_EXPIRE"))

if cache_purge_cfg != "" and not cache_no_expire:  # coverage: ignore
    cache_purge = pd.Timedelta(cache_purge_cfg)
    now = pd.Timestamp("now").timestamp()

    purgeable = list(
        path
        for path in cache_dir.glob("opensky/*")
        if now - path.lstat().st_mtime > cache_purge.total_seconds()
    )

    if len(purgeable) > 0:
        _log.warn(
            f"Removing {len(purgeable)} cache files older than {cache_purge}"
        )
        for path in purgeable:
            path.unlink()

if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

# -- Tqdm Style Configuration --
tqdm_style = config.get("global", "tqdm_style", fallback="auto")
_log.info(f"Selected tqdm style: {tqdm_style}")

# -- Plugin management --

_enabled_plugins_raw = config.get("plugins", "enabled_plugins", fallback="")
_enabled_list = ",".join(_enabled_plugins_raw.split("\n")).split(",")

_selected = set(s.replace("-", "").strip().lower() for s in _enabled_list)
_selected -= {""}

_log.info(f"Selected plugins: {_selected}")

if "TRAFFIC_NOPLUGIN" not in os.environ.keys():  # coverage: ignore
    ep: Iterable[EntryPoint]
    try:
        # https://docs.python.org/3/library/importlib.metadata.html#entry-points
        ep = entry_points(group="traffic.plugins")
    except TypeError:
        ep = entry_points().get("traffic.plugins", [])
    for entry_point in ep:
        name = entry_point.name.replace("-", "").lower()
        if name in _selected:
            _log.info(f"Loading plugin: {name}")
            handle = entry_point.load()
            _log.info(f"Loaded plugin: {handle.__name__}")
            load = getattr(handle, "_onload", None)
            if load is not None:
                load()
