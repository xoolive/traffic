import configparser
from pathlib import Path

from appdirs import user_cache_dir, user_config_dir

from .adsb.flightradar24 import FlightRadar24
from .adsb.opensky import OpenSky

from .airac import SectorParser
from .aircraft import Aircraft
from .airport import AirportParser
from .airways import Airways
from .navaid import NavaidParser

# Parse configuration and input specific parameters in below classes

__all__ = ['aircraft', 'airports', 'airways', 'navaids', 'sectors',
           'flightradar24', 'opensky', ]

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
config.read(config_file.as_posix())

airac_path_str = config.get("global", "airac_path", fallback="")
if airac_path_str != "":
    SectorParser.airac_path = Path(airac_path_str)

SectorParser.cache_dir = cache_dir
FlightRadar24.username = config.get("global", "fr24_username", fallback="")
FlightRadar24.password = config.get("global", "fr24_password", fallback="")
FlightRadar24.cache_dir = cache_dir / "flightradar24"

AirportParser.cache = cache_dir / "airports.pkl"
NavaidParser.cache = cache_dir / "navaids.pkl"
Airways.cache = cache_dir / "airways.pkl"
Aircraft.cache = cache_dir / "aircraft.pkl"

OpenSky.cache_dir = cache_dir / "impala"
# not great here...
if not OpenSky.cache_dir.exists():
    OpenSky.cache_dir.mkdir(parents=True)

sectors = SectorParser(config_file)
airports = AirportParser()
navaids = NavaidParser()
flightradar24 = FlightRadar24()
airways = Airways()
aircraft = Aircraft()

opensky_username = config.get("global", "opensky_username", fallback="")
opensky_password = config.get("global", "opensky_password", fallback="")
opensky = OpenSky(opensky_username, opensky_password)
