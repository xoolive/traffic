import logging


def log_mode(mode: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, mode))
