import logging

logger: logging.Logger = logging.getLogger(__name__)

logger.warning(
    "Prefer `from traffic.core import loglevel`", DeprecationWarning
)


def loglevel(mode: str) -> None:
    logger_traffic = logging.getLogger("traffic")
    logger_traffic.setLevel(getattr(logging, mode))
