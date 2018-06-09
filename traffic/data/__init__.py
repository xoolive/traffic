import configparser
from pathlib import Path

from appdirs import user_cache_dir, user_config_dir

from .adsb.flightradar24 import FlightRadar24
from .adsb.opensky import OpenSky

from .sectors.airac import SectorParser
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
        # TODO prepare a bit more
        fh.write("[global]\nairac_path = ")

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
sectors = SectorParser(config_file)

opensky_username = config.get("global", "opensky_username", fallback="")
opensky_password = config.get("global", "opensky_password", fallback="")
opensky = OpenSky(opensky_username, opensky_password, cache_dir / "opensky")

fr24_username = config.get("global", "fr24_username", fallback="")
fr24_password = config.get("global", "fr24_password", fallback="")
flightradar24 = FlightRadar24(cache_dir / "flightradar24",
                              fr24_username, fr24_password)
