import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from httpx import Client

from .. import cache_dir, config, config_file

if TYPE_CHECKING:
    from .adsb.opensky import OpenSky
    from .basic.aircraft import Aircraft
    from .basic.airports import Airports
    from .basic.airways import Airways
    from .basic.navaid import Navaids
    from .basic.runways import Runways
    from .eurocontrol.aixm.airports import AIXMAirportParser
    from .eurocontrol.aixm.airspaces import AIXMAirspaceParser
    from .eurocontrol.aixm.navpoints import AIXMNavaidParser
    from .eurocontrol.aixm.routes import AIXMRoutesParser
    from .eurocontrol.ddr.airspaces import NMAirspaceParser
    from .eurocontrol.ddr.allft import AllFT
    from .eurocontrol.ddr.navpoints import NMNavaids
    from .eurocontrol.ddr.routes import NMRoutes
    from .eurocontrol.eurofirs import Eurofirs

# Parse configuration and input specific parameters in below classes

__all__ = [
    "aircraft",
    "airports",
    "airways",
    "navaids",
    "runways",
    "aixm_airports",
    "aixm_airspaces",
    "aixm_airways",
    "aixm_navaids",
    "nm_airspaces",
    "nm_airways",
    "nm_freeroute",
    "nm_navaids",
    "eurofirs",
    "opensky",
    "client",
    "AllFT",
    "Navaids",
]

aircraft: "Aircraft"
airports: "Airports"
airways: "Airways"
navaids: "Navaids"
runways: "Runways"
aixm_airports: "AIXMAirportParser"
aixm_airspaces: "AIXMAirspaceParser"
aixm_airways: "AIXMRoutesParser"
aixm_navaids: "AIXMNavaidParser"
nm_airspaces: "NMAirspaceParser"
nm_airways: "NMRoutes"
nm_navaids: "NMNavaids"
eurofirs: "Eurofirs"
opensky: "OpenSky"
client: Client


aixm_path_str = config.get("global", "aixm_path", fallback="")
nm_path_str = config.get("global", "nm_path", fallback="")

# We keep "" for forcing to no proxy

http_proxy = config.get("network", "http.proxy", fallback="<>")
https_proxy = config.get("network", "https.proxy", fallback="<>")
paramiko_proxy = config.get("network", "ssh.proxycommand", fallback="")

proxy_values = dict(
    (key, value)
    for key, value in [("http", http_proxy), ("https", https_proxy)]
    if value != "<>"
)


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

    if name == "aixm_airways":  # coverage: ignore
        from .eurocontrol.aixm.routes import AIXMRoutesParser

        AIXMRoutesParser.cache_dir = cache_dir
        res = AIXMRoutesParser.from_file(Path(aixm_path_str))
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
        from .adsb.opensky import OpenSky

        res = OpenSky()
        _cached_imports[name] = res
        return res

    if name == "runways":
        from .basic.runways import Runways

        Runways.cache_dir = cache_dir
        res = Runways()
        _cached_imports[name] = res
        return res

    if name == "client":
        res = Client(follow_redirects=True)
        _cached_imports[name] = res
        return res

    if name == "nm_b2b":
        raise NotImplementedError(
            "This has been deprecated from traffic."
            "Please install optional library pyb2b instead."
        )

    if name == "AllFT":
        from .eurocontrol.ddr.allft import AllFT

        return AllFT

    raise AttributeError(f"module {__name__} has no attribute {name}")
