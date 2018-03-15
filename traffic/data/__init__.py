from .airac import SectorParser
from .flightradar24 import FlightRadar24

# Parse configuration and input specific parameters in below classes

import configparser
from appdirs import user_config_dir, user_cache_dir
from pathlib import Path

config_dir = Path(user_config_dir("traffic"))
cache_dir = Path(user_cache_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():
    config_dir.mkdir()
    with config_file.open('w') as fh:
        fh.write("[global]\nairac_path = ")

if not cache_dir.exists():
    cache_dir.mkdir()

config = configparser.ConfigParser()
config.read(config_file)

airac_path_str = config.get("global", "airac_path", fallback="")
if airac_path_str != "":
    SectorParser.airac_path = Path(airac_path_str)

SectorParser.cache_dir = cache_dir

sectors = SectorParser(config_file)

FlightRadar24.token = config.get("global", "fr24_token", fallback="")
