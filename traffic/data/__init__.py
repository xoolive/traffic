import configparser
from pathlib import Path

from appdirs import user_cache_dir, user_config_dir

from .adsb.opensky import OpenSky
from .adsb import sqlite  # this could be a plugin...

from .sectors.airac import SectorParser
from .sectors.eurofirs import eurofirs
from .basic.aircraft import Aircraft
from .basic.airport import AirportParser
from .basic.airways import Airways
from .basic.navaid import NavaidParser

# Parse configuration and input specific parameters in below classes

__all__ = ['aircraft', 'airports', 'airways', 'navaids', 'sectors',
           'flightradar24', 'opensky', ]

config_dir = Path(user_config_dir("traffic"))
cache_dir = Path(user_cache_dir("traffic"))
config_file = config_dir / "traffic.conf"

if not config_dir.exists():
    config_dir.mkdir()
    with config_file.open('w') as fh:
        fh.write("""[global]
airac_path =
opensky_username =
opensky_password =
""")

if not cache_dir.exists():
    cache_dir.mkdir()

config = configparser.ConfigParser()
config.read(config_file.as_posix())

airac_path_str = config.get("global", "airac_path", fallback="")
if airac_path_str != "":
    SectorParser.airac_path = Path(airac_path_str)

Aircraft.cache = cache_dir / "aircraft.pkl"
AirportParser.cache = cache_dir / "airports.pkl"
Airways.cache = cache_dir / "airways.pkl"
NavaidParser.cache = cache_dir / "navaids.pkl"
SectorParser.cache_dir = cache_dir

aircraft = Aircraft()
airports = AirportParser()
airways = Airways()
navaids = NavaidParser()
airac = SectorParser(config_file)
sectors = airac  # deprecated?

opensky_username = config.get("global", "opensky_username", fallback="")
opensky_password = config.get("global", "opensky_password", fallback="")
opensky = OpenSky(opensky_username, opensky_password, cache_dir / "opensky")
