import logging


def loglevel(mode: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, mode))
