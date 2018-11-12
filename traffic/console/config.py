import argparse
import logging

from . import dispatch_open


def main(args):

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

    args = parser.parse_args(args)

    if args.list:
        print(config_dir)

    if args.edit:
        logging.info("Open configuration file {}".format(config_file))
        dispatch_open(config_file)

    if args.open:
        logging.info("Open configuration directory {}".format(config_file))
        dispatch_open(config_dir)
