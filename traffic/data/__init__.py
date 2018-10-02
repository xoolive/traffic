from pathlib import Path

from .. import config, cache_dir, config_file
from .adsb.opensky import OpenSky
from .adsb.decode import Decoder as ModeS_Decoder  # noqa: F401
from .airspaces.airac import AirspaceParser
from .airspaces.eurofirs import eurofirs
from .basic.aircraft import Aircraft
from .basic.airport import AirportParser
from .basic.airways import Airways
from .basic.navaid import NavaidParser
from .basic.runways import Runways
from .so6 import SO6  # noqa: F401

# Parse configuration and input specific parameters in below classes

__all__ = ['aircraft', 'airports', 'airways', 'navaids', 'airac', 'eurofirs',
           'opensky', ]

airac_path_str = config.get("global", "airac_path", fallback="")
if airac_path_str != "":
    AirspaceParser.airac_path = Path(airac_path_str)

Aircraft.cache = cache_dir / "aircraft.pkl"
AirportParser.cache = cache_dir / "airports.pkl"
Airways.cache = cache_dir / "airways.pkl"
NavaidParser.cache = cache_dir / "navaids.pkl"
Runways.cache = cache_dir / "runways.pkl"
AirspaceParser.cache_dir = cache_dir

aircraft = Aircraft()
airports = AirportParser()
airways = Airways()
navaids = NavaidParser()
airac = AirspaceParser(config_file)
runways = Runways().runways

opensky_username = config.get("global", "opensky_username", fallback="")
opensky_password = config.get("global", "opensky_password", fallback="")
opensky = OpenSky(opensky_username, opensky_password, cache_dir / "opensky")
