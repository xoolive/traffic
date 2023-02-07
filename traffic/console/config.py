import argparse
import logging
from typing import List

from . import dispatch_open

logger = logging.getLogger(__name__)


def main(args_list: List[str]) -> None:
    from .. import config_dir, config_file

    parser = argparse.ArgumentParser(
        prog="traffic config",
        description="traffic configuration file and directory",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list",
        "-l",
        dest="list",
        action="store_true",
        help="print the path of the configuration directory",
    )
    group.add_argument(
        "--edit",
        "-e",
        dest="edit",
        action="store_true",
        help="open the configuration file for edition",
    )
    group.add_argument(
        "--open",
        "-o",
        dest="open",
        action="store_true",
        help="open the configuration directory in your native file browser",
    )

    args = parser.parse_args(args_list)

    if args.list:
        print(config_dir)

    if args.edit:
        logger.info("Open configuration file {}".format(config_file))
        dispatch_open(config_file)

    if args.open:
        logger.info("Open configuration directory {}".format(config_file))
        dispatch_open(config_dir)
