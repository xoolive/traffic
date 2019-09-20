import logging
import sys
from functools import lru_cache
from pathlib import Path

from .. import cache_dir, config, config_file
from .adsb.decode import Decoder as ModeS_Decoder  # noqa: F401
from .adsb.opensky import OpenSky
from .airspaces.eurocontrol_aixm import AIXMAirspaceParser
from .airspaces.eurofirs import eurofirs
from .basic.aircraft import Aircraft
from .basic.airports import Airports
from .basic.airways import Airways
from .basic.navaid import Navaids
from .basic.runways import Runways
from .eurocontrol.ddr.airspaces import NMAirspaceParser
from .eurocontrol.ddr.navpoints import NMNavaids
from .eurocontrol.ddr.routes import NMRoutes
from .eurocontrol.ddr.so6 import SO6  # noqa: F401
from .eurocontrol.nmb2b import NMB2B

# Parse configuration and input specific parameters in below classes

__all__ = [
    "aircraft",
    "airports",
    "airways",
    "navaids",
    "aixm_airspaces",
    "nm_airspaces",
    "nm_airways",
    "nm_navaids",
    "eurofirs",
    "opensky",
]

airac_path_str = config.get("global", "airac_path", fallback="")
if airac_path_str != "":  # coverage: ignore
    logging.warning(
        "Rename airac_path to aixm_path in your configuration file. "
        "The old name will not be supported in future versions"
    )
    AIXMAirspaceParser.aixm_path = Path(airac_path_str)

aixm_path_str = config.get("global", "aixm_path", fallback="")
if aixm_path_str != "":  # coverage: ignore
    AIXMAirspaceParser.aixm_path = Path(aixm_path_str)

nm_path_str = config.get("global", "nm_path", fallback="")
if nm_path_str != "":  # coverage: ignore
    NMAirspaceParser.nm_path = Path(nm_path_str)
    NMRoutes.nm_path = Path(nm_path_str)

Aircraft.cache_dir = cache_dir
Airports.cache_dir = cache_dir
Airways.cache_dir = cache_dir
Navaids.cache_dir = cache_dir
Runways.cache_dir = cache_dir
AIXMAirspaceParser.cache_dir = cache_dir

opensky_username = config.get("global", "opensky_username", fallback="")
opensky_password = config.get("global", "opensky_password", fallback="")

# We keep "" for forcing to no proxy
http_proxy = config.get("proxy", "http", fallback="<>")
https_proxy = config.get("proxy", "https", fallback="<>")

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


if sys.version_info < (3, 7, 0):
    aircraft = Aircraft()
    airports = Airports()
    airways = Airways()
    navaids = Navaids()
    runways = Runways()

    aixm_airspaces = AIXMAirspaceParser(config_file)
    nm_airspaces = NMAirspaceParser(config_file)
    nm_navaids = NMNavaids.from_file(nm_path_str)
    nm_airways = NMRoutes()

    opensky = OpenSky(opensky_username, opensky_password, cache_dir / "opensky")
    if len(proxy_values) > 0:
        opensky.session.proxies.update(proxy_values)
        opensky.session.trust_env = False

    if pkcs12_filename != "" and pkcs12_password != "":
        logging.debug(f"pcks12_filename: {pkcs12_filename}")
        nmb2b = NMB2B(
            getattr(NMB2B, nmb2b_mode),
            nmb2b_version,
            pkcs12_filename,
            pkcs12_password,
        )
        if len(proxy_values) > 0:
            nmb2b.session.proxies.update(proxy_values)
            nmb2b.session.trust_env = False


@lru_cache()
def __getattr__(name: str):
    """This only works for Python >= 3.7, see PEP 562."""
    if name == "aircraft":
        return Aircraft()
    if name == "airports":
        return Airports()
    if name == "airways":
        return Airways()
    if name == "navaids":
        return Navaids()
    if name == "aixm_airspaces":  # coverage: ignore
        return AIXMAirspaceParser(config_file)
    if name == "nm_airspaces":  # coverage: ignore
        return NMAirspaceParser(config_file)
    if name == "nm_navaids":  # coverage: ignore
        return NMNavaids.from_file(nm_path_str)
    if name == "nm_airways":
        return NMRoutes()
    if name == "opensky":
        opensky = OpenSky(
            opensky_username, opensky_password, cache_dir / "opensky"
        )
        if len(proxy_values) > 0:
            opensky.session.proxies.update(proxy_values)
            opensky.session.trust_env = False
        return opensky
    if name == "runways":
        return Runways()
    if name == "nmb2b":
        if pkcs12_filename != "" and pkcs12_password != "":
            logging.debug(f"pcks12_filename: {pkcs12_filename}")
            nmb2b = NMB2B(
                getattr(NMB2B, nmb2b_mode),
                nmb2b_version,
                pkcs12_filename,
                pkcs12_password,
            )
            if len(proxy_values) > 0:
                nmb2b.session.proxies.update(proxy_values)
                nmb2b.session.trust_env = False
            return nmb2b
    if name == "airac":  # coverage: ignore
        cache_file = cache_dir / "airac.cache"
        if cache_file.exists():
            cache_file.unlink()
        logging.warning(
            f"""DEPRECATION WARNING. Please note that:
            - airac has been renamed into aixm_airspaces.
            - backward compatibility will be removed in future versions.
            """
        )
        return AIXMAirspaceParser(config_file)

    raise AttributeError(f"module {__name__} has no attribute {name}")
