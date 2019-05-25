import configparser
import logging
import os
import warnings
from pathlib import Path

import pkg_resources
from appdirs import user_cache_dir, user_config_dir
from tqdm import TqdmExperimentalWarning

# Silence this warning about autonotebook mode for tqdm
warnings.simplefilter("ignore", TqdmExperimentalWarning)

# -- Configuration management --

config_dir = Path(user_config_dir("traffic"))
cache_dir = Path(user_cache_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():
    config_dir.mkdir(parents=True)
    with config_file.open("w") as fh:
        fh.write(
            f"""[global]
nm_path =
opensky_username =
opensky_password =
[plugins]
enabled_plugins = CesiumJS, Leaflet
"""
        )

if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

config = configparser.ConfigParser()
config.read(config_file.as_posix())

# -- Plugin management --

_selected = [
    s.strip().lower()
    for s in config.get("plugins", "enabled_plugins", fallback="").split(",")
]

logging.info(f"Selected plugins: {_selected}")

if "TRAFFIC_NOPLUGIN" not in os.environ.keys():  # coverage: ignore
    for entry_point in pkg_resources.iter_entry_points("traffic.plugins"):
        if entry_point.name.lower() in _selected:
            handle = entry_point.load()
            logging.info(f"Loading plugin: {handle.__name__}")
            load = getattr(handle, "_onload", None)
            if load is not None:
                load()
