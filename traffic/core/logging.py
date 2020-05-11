import logging

logging.warning(
    "Prefer `from traffic.core import loglevel`", DeprecationWarning
)


def loglevel(mode: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, mode))
