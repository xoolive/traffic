import configparser
import imp
import inspect
import logging
import warnings
from pathlib import Path

from appdirs import user_cache_dir, user_config_dir
from tqdm import TqdmExperimentalWarning

from .plugins import PluginProvider

# Silence this warning about autonotebook mode for tqdm
warnings.simplefilter('ignore', TqdmExperimentalWarning)

# -- Configuration management --

config_dir = Path(user_config_dir("traffic"))
cache_dir = Path(user_cache_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():
    config_dir.mkdir(parents=True)
    with config_file.open("w") as fh:
        fh.write(
            f"""[global]
airac_path =
opensky_username =
opensky_password =
[plugins]
enabled_plugins = Bluesky, CesiumJS
"""
        )

if not cache_dir.exists():
    cache_dir.mkdir(parents=True)

config = configparser.ConfigParser()
config.read(config_file.as_posix())

# -- Plugin management --

_selected = [
    s.strip()
    for s in config.get("plugins", "enabled_plugins", fallback="").split(",")
]
_plugin_paths = [Path(__file__).parent / "plugins", config_dir / "plugins"]
_all_plugins = []

for path in _plugin_paths:
    for f in path.glob("[a-zA-Z]*.py"):
        a = imp.load_source(f.stem, f.as_posix())
        for name, cls in inspect.getmembers(a, inspect.isclass):
            if PluginProvider in cls.__mro__:
                _all_plugins.append(cls())

for plugin in _all_plugins:
    if plugin.title in _selected:
        logging.info(f"Loading {plugin.title}")
        plugin.load_plugin()
