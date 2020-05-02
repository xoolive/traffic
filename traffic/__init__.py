import configparser
import logging
import os
import warnings
from pathlib import Path

# importing ipyleaflet here avoids annoying warnings
import ipyleaflet  # noqa: F401
import pandas as pd
import pkg_resources
from appdirs import user_cache_dir, user_config_dir

from tqdm import TqdmExperimentalWarning

# Silence this warning about autonotebook mode for tqdm
warnings.simplefilter("ignore", TqdmExperimentalWarning)

# -- Configuration management --

config_dir = Path(user_config_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():
    config_template = (Path(__file__).parent / "traffic.conf").read_text()
    config_dir.mkdir(parents=True)
    config_file.write_text(config_template)

config = configparser.ConfigParser()
config.read(config_file.as_posix())

# Check the config file for a cache directory. If not present
# then use the system default cache path

cache_dir = Path(user_cache_dir("traffic"))

cache_dir_cfg = config.get("global", "cache_dir", fallback="").strip()
if cache_dir_cfg != "":
    warnings.warn(
        """Please edit your configuration file:

        # Old style, will soon no longer be supported
        [global]
        cache_dir =

        # New style, with extra parameters

        [cache]
        # path =

        ## number of days, databases will be downloaded again
        # expiration = 180 days # default value

        ## Save your disk space!
        ## Impala cache files will be removed after a given number of days
        ## By default, cache files are left untouched.
        # purge =
        """,
        DeprecationWarning,
    )
    cache_dir = Path(cache_dir_cfg)

cache_dir_cfg = config.get("cache", "path", fallback="").strip()
if cache_dir_cfg != "":
    cache_dir = Path(cache_dir_cfg)

cache_expiration_cfg = config.get("cache", "expiration", fallback="180 days")
cache_expiration = pd.Timedelta(cache_expiration_cfg)

cache_purge_cfg = config.get("cache", "purge", fallback="")
if cache_purge_cfg != "":
    cache_purge = pd.Timedelta(cache_purge_cfg)

    purgeable = list(
        path
        for path in cache_dir.glob("opensky/*")
        if pd.Timestamp("now") - pd.Timestamp(path.lstat().st_mtime * 1e9)
        > cache_purge
    )

    if len(purgeable) > 0:
        logging.warn(
            f"Removing {len(purgeable)} cache files older than {cache_purge}"
        )
        for path in purgeable:
            path.unlink()


if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

# -- Plugin management --

_enabled_plugins_raw = config.get("plugins", "enabled_plugins", fallback="")
_enabled_list = ",".join(_enabled_plugins_raw.split("\n")).split(",")

_selected = set(s.replace("-", "").strip().lower() for s in _enabled_list)
_selected -= {""}

logging.info(f"Selected plugins: {_selected}")

if "TRAFFIC_NOPLUGIN" not in os.environ.keys():  # coverage: ignore
    for entry_point in pkg_resources.iter_entry_points("traffic.plugins"):
        if entry_point.name.replace("-", "").lower() in _selected:
            handle = entry_point.load()
            logging.info(f"Loading plugin: {handle.__name__}")
            load = getattr(handle, "_onload", None)
            if load is not None:
                load()
