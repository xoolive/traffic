from .airac import get_area, init_airac  # noqa

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
    raise ImportError(f"Please edit file {config_file} with AIRAC directory")

if not cache_dir.exists():
    cache_dir.mkdir()

config = configparser.ConfigParser()
config.read(config_file)

airac_path_str = config.get("global", "airac_path", fallback="")
if airac_path_str == "":
    raise ImportError(f"Please edit file {config_file} with AIRAC directory")

airac_path = Path(airac_path_str)
if not airac_path.exists():
    raise ImportError(f"Please edit file {config_file} with AIRAC directory")

init_airac(airac_path, cache_dir)
