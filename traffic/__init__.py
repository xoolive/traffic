import configparser
import imp
import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path

from appdirs import user_cache_dir, user_config_dir

from .plugins import PluginProvider

config_dir = Path(user_config_dir("traffic"))
cache_dir = Path(user_cache_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():
    config_dir.mkdir()
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
    cache_dir.mkdir()

config = configparser.ConfigParser()
config.read(config_file.as_posix())


def edit_config():
    if sys.platform.startswith("darwin"):
        subprocess.call(("open", config_file))
    elif os.name == "nt":  # For Windows
        os.startfile(config_dir)
    elif os.name == "posix":  # For Linux, Mac, etc.
        subprocess.call(("xdg-open", config_file))


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
