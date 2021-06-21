import logging
from pathlib import Path
from typing import Any, Dict

from .. import cache_dir, config, config_file

# Parse configuration and input specific parameters in below classes

__all__ = [
    "aircraft",
    "airports",
    "airways",
    "carto_session",
    "navaids",
    "aixm_airspaces",
    "nm_airspaces",
    "nm_airways",
    "nm_navaids",
    "eurofirs",
    "opensky",
    "session",
    "AllFT",
    "ModeS_Decoder",
    "SO6",
]


aixm_path_str = config.get("global", "aixm_path", fallback="")
nm_path_str = config.get("global", "nm_path", fallback="")

opensky_username = config.get("opensky", "username", fallback="")
opensky_password = config.get("opensky", "password", fallback="")

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
nmb2b_version = config.get("nmb2b", "version", fallback="23.0.0")


_cached_imports: Dict[str, Any] = dict()


def __getattr__(name: str):

    res: Any
    if name in _cached_imports.keys():
        return _cached_imports[name]

    """This only works for Python >= 3.7, see PEP 562."""
    if name == "aircraft":
        from .basic.aircraft import Aircraft

        Aircraft.cache_dir = cache_dir
        res = Aircraft()
        _cached_imports[name] = res
        return res

    if name == "airports_fr24":
        from .basic.airports import Airports

        Airports.cache_dir = cache_dir
        res = Airports(src="fr24")
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
        from .airspaces.eurofirs import eurofirs

        return eurofirs

    if name == "navaids":
        from .basic.navaid import Navaids

        Navaids.cache_dir = cache_dir
        res = Navaids()
        _cached_imports[name] = res
        return res

    if name == "aixm_airspaces":  # coverage: ignore
        from .airspaces.eurocontrol_aixm import AIXMAirspaceParser

        AIXMAirspaceParser.cache_dir = cache_dir

        if aixm_path_str != "":  # coverage: ignore
            AIXMAirspaceParser.aixm_path = Path(aixm_path_str)

        res = AIXMAirspaceParser(config_file)
        _cached_imports[name] = res
        return res

    if name == "nm_airspaces":  # coverage: ignore
        from .eurocontrol.ddr.airspaces import NMAirspaceParser

        if nm_path_str != "":  # coverage: ignore
            NMAirspaceParser.nm_path = Path(nm_path_str)
        res = NMAirspaceParser(config_file)
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

        opensky = OpenSky(
            opensky_username,
            opensky_password,
            cache_dir / "opensky",
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
        from requests import Session

        session = Session()
        if len(proxy_values) > 0:
            session.proxies.update(proxy_values)
            session.trust_env = False
        res = session
        _cached_imports[name] = res
        return res

    if name == "nm_b2b":
        from .eurocontrol.b2b import NMB2B

        if pkcs12_filename != "" and pkcs12_password != "":
            logging.debug(f"pcks12_filename: {pkcs12_filename}")
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
        from .adsb.decode import Decoder as ModeS_Decoder

        return ModeS_Decoder

    if name == "SO6":
        from .eurocontrol.ddr.so6 import SO6

        return SO6

    raise AttributeError(f"module {__name__} has no attribute {name}")
