import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from requests import Session

from .. import cache_dir, config, config_file

if TYPE_CHECKING:
    from .adsb.decode import ModeS_Decoder as ModeS_Decoder
    from .adsb.opensky import OpenSky
    from .basic.aircraft import Aircraft
    from .basic.airports import Airports
    from .basic.airways import Airways
    from .basic.navaid import Navaids
    from .basic.runways import Runways
    from .eurocontrol.aixm.airports import AIXMAirportParser
    from .eurocontrol.aixm.airspaces import AIXMAirspaceParser
    from .eurocontrol.aixm.navpoints import AIXMNavaidParser
    from .eurocontrol.b2b import NMB2B
    from .eurocontrol.ddr.airspaces import NMAirspaceParser
    from .eurocontrol.ddr.allft import AllFT
    from .eurocontrol.ddr.navpoints import NMNavaids
    from .eurocontrol.ddr.routes import NMRoutes
    from .eurocontrol.ddr.so6 import SO6
    from .eurocontrol.eurofirs import Eurofirs

# Parse configuration and input specific parameters in below classes

__all__ = [
    "aircraft",
    "airports",
    "airways",
    "carto_session",
    "navaids",
    "runways",
    "aixm_airports",
    "aixm_airspaces",
    "aixm_navaids",
    "nm_airspaces",
    "nm_airways",
    "nm_freeroute",
    "nm_navaids",
    "eurofirs",
    "opensky",
    "session",
    "AllFT",
    "ModeS_Decoder",
    "Navaids",
    "SO6",
]

aircraft: "Aircraft"
airports: "Airports"
airways: "Airways"
carto_session: Session
navaids: "Navaids"
runways: "Runways"
aixm_airports: "AIXMAirportParser"
aixm_airspaces: "AIXMAirspaceParser"
aixm_navaids: "AIXMNavaidParser"
nm_airspaces: "NMAirspaceParser"
nm_airways: "NMRoutes"
nm_b2b: "NMB2B"
nm_navaids: "NMNavaids"
eurofirs: "Eurofirs"
opensky: "OpenSky"
session: Session


aixm_path_str = config.get("global", "aixm_path", fallback="")
nm_path_str = config.get("global", "nm_path", fallback="")

# Give priority to environment variables
opensky_username = os.environ.get(
    "OPENSKY_USERNAME", config.get("opensky", "username", fallback="")
)
opensky_password = os.environ.get(
    "OPENSKY_PASSWORD", config.get("opensky", "password", fallback="")
)


# We keep "" for forcing to no proxy

http_proxy = config.get("network", "http.proxy", fallback="<>")
https_proxy = config.get("network", "https.proxy", fallback="<>")
paramiko_proxy = config.get("network", "ssh.proxycommand", fallback="")

proxy_values = dict(
    (key, value)
    for key, value in [("http", http_proxy), ("https", https_proxy)]
    if value != "<>"
)


pkcs12_filename = config.get("nmb2b", "pkcs12_filename", fallback="")
pkcs12_password = config.get("nmb2b", "pkcs12_password", fallback="")
nmb2b_mode = config.get("nmb2b", "mode", fallback="PREOPS")
if nmb2b_mode not in ["OPS", "PREOPS"]:
    raise RuntimeError("mode must be one of OPS or PREOPS")
nmb2b_version = config.get("nmb2b", "version", fallback="25.0.0")


_cached_imports: Dict[str, Any] = dict()

_log = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:

    res: Any
    if name in _cached_imports.keys():
        return _cached_imports[name]

    if name == "aircraft":
        from .basic.aircraft import Aircraft

        Aircraft.cache_dir = cache_dir

        filename = config.get("aircraft", "database", fallback="")
        if filename != "":
            res = Aircraft.from_file(filename)
            rename_cols = dict(
                (value, key)  # yes, reversed...
                for key, value in config["aircraft"].items()
                if key != "database"
            )
            if len(rename_cols) > 0:
                res = res.rename(columns=rename_cols)
        else:
            res = Aircraft()
        _cached_imports[name] = res.fillna("")
        return res.fillna("")

    if name == "airports_fr24":
        from .basic.airports import Airports

        Airports.cache_dir = cache_dir
        res = Airports()
        res._src = "fr24"
        _cached_imports[name] = res
        return res

    if name == "airports":
        from .basic.airports import Airports

        Airports.cache_dir = cache_dir
        res = Airports()
        _cached_imports[name] = res
        return res

    if name == "airways":
        from .basic.airways import Airways

        Airways.cache_dir = cache_dir
        res = Airways()
        _cached_imports[name] = res
        return res

    if name == "carto_session":
        from cartes.osm.requests import session as carto_session

        if len(proxy_values) > 0:
            carto_session.proxies.update(proxy_values)
            carto_session.trust_env = False

        res = carto_session
        _cached_imports[name] = res
        return res

    if name == "eurofirs":
        from .eurocontrol.eurofirs import eurofirs

        return eurofirs

    if name == "navaids":
        from .basic.navaid import Navaids

        Navaids.cache_dir = cache_dir
        res = Navaids()
        _cached_imports[name] = res
        return res

    if name == "aixm_navaids":  # coverage: ignore
        from .eurocontrol.aixm.navpoints import AIXMNavaidParser

        AIXMNavaidParser.cache_dir = cache_dir
        res = AIXMNavaidParser.from_file(Path(aixm_path_str))
        _cached_imports[name] = res
        return res

    if name == "aixm_airports":  # coverage: ignore
        from .eurocontrol.aixm.airports import AIXMAirportParser

        AIXMAirportParser.cache_dir = cache_dir
        res = AIXMAirportParser(
            data=None,
            aixm_path=Path(aixm_path_str)
            if aixm_path_str is not None
            else None,
        )
        _cached_imports[name] = res
        return res

    if name == "aixm_airspaces":  # coverage: ignore
        from .eurocontrol.aixm.airspaces import AIXMAirspaceParser

        AIXMAirspaceParser.cache_dir = cache_dir
        res = AIXMAirspaceParser(
            data=None,
            aixm_path=Path(aixm_path_str)
            if aixm_path_str is not None
            else None,
        )
        _cached_imports[name] = res
        return res

    if name == "nm_airspaces":  # coverage: ignore
        from .eurocontrol.ddr.airspaces import NMAirspaceParser

        if nm_path_str != "":  # coverage: ignore
            NMAirspaceParser.nm_path = Path(nm_path_str)
        res = NMAirspaceParser(data=None, config_file=config_file)
        _cached_imports[name] = res
        return res

    if name == "nm_freeroute":  # coverage: ignore
        from .eurocontrol.ddr.freeroute import NMFreeRouteParser

        if nm_path_str != "":  # coverage: ignore
            NMFreeRouteParser.nm_path = Path(nm_path_str)
        res = NMFreeRouteParser(data=None, config_file=config_file)
        _cached_imports[name] = res
        return res

    if name == "nm_navaids":  # coverage: ignore
        from .eurocontrol.ddr.navpoints import NMNavaids

        res = NMNavaids.from_file(nm_path_str)
        _cached_imports[name] = res
        return res

    if name == "nm_airways":
        from .eurocontrol.ddr.routes import NMRoutes

        if nm_path_str != "":  # coverage: ignore
            NMRoutes.nm_path = Path(nm_path_str)
        res = NMRoutes()
        _cached_imports[name] = res
        return res

    if name == "opensky":
        from . import session
        from .adsb.opensky import OpenSky

        # give priority to the OPENSKY_CACHE environment variable
        opensky_cache = os.environ.get("OPENSKY_CACHE", None)
        if opensky_cache is not None:
            opensky_cache_path = Path(opensky_cache)
        else:
            opensky_cache_path = cache_dir / "opensky"
        if not opensky_cache_path.exists():
            opensky_cache_path = cache_dir / "opensky"

        opensky = OpenSky(
            opensky_username,
            opensky_password,
            opensky_cache_path,
            session,
            paramiko_proxy,
        )
        res = opensky
        _cached_imports[name] = res
        return res

    if name == "runways":
        from .basic.runways import Runways

        Runways.cache_dir = cache_dir
        res = Runways()
        _cached_imports[name] = res
        return res

    if name == "session":

        session = Session()
        if len(proxy_values) > 0:
            session.proxies.update(proxy_values)
            session.trust_env = False
        res = session
        _cached_imports[name] = res
        return res

    if name == "nm_b2b":
        from . import session
        from .eurocontrol.b2b import NMB2B

        if pkcs12_filename != "" and pkcs12_password != "":
            _log.debug(f"pcks12_filename: {pkcs12_filename}")
            nm_b2b = NMB2B(
                getattr(NMB2B, nmb2b_mode),
                nmb2b_version,
                session,
                pkcs12_filename,
                pkcs12_password,
            )
            res = nm_b2b
            _cached_imports[name] = res
            return res

    if name == "AllFT":
        from .eurocontrol.ddr.allft import AllFT

        return AllFT

    if name == "ModeS_Decoder":
        from .adsb.decode import ModeS_Decoder

        return ModeS_Decoder

    if name == "SO6":
        from .eurocontrol.ddr.so6 import SO6

        return SO6

    raise AttributeError(f"module {__name__} has no attribute {name}")
