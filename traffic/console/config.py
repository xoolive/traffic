import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


def dispatch_open(filename: Path):
    if sys.platform.startswith("darwin"):
        subprocess.call(("open", filename))
    elif os.name == "nt":  # For Windows
        os.startfile(filename)
    elif os.name == "posix":  # For Linux, Mac, etc.
        subprocess.call(("xdg-open", filename))


def main(args):

    from .. import cache_dir, config_dir, config_file

    parser = argparse.ArgumentParser(
        prog="traffic config",
        description="traffic configuration file and directory",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--print",
        "-p",
        dest="print",
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
        "--cache",
        "-c",
        dest="cache",
        action="store_true",
        help="open the cache directory in your native file browser",
    )
    group.add_argument(
        "--dir",
        "-d",
        dest="dir",
        action="store_true",
        help="open the configuration directory in your native file browser",
    )

    args = parser.parse_args(args)

    if args.print:
        print(config_dir)

    if args.edit:
        logging.info("Open configuration file {}".format(config_file))
        dispatch_open(config_file)

    if args.cache:
        logging.info("Open cache directory {}".format(cache_dir))
        dispatch_open(cache_dir)

    if args.dir:
        logging.info("Open configuration directory {}".format(config_file))
        dispatch_open(config_dir)
