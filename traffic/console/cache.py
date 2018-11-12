import argparse
import logging

from . import dispatch_open


def main(args):

    from .. import cache_dir

    parser = argparse.ArgumentParser(
        prog="traffic cache", description="traffic cache directory"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list",
        "-l",
        dest="list",
        action="store_true",
        help="print the path of the cache directory",
    )
    group.add_argument(
        "--open",
        "-o",
        dest="open",
        action="store_true",
        help="open the cache directory in your native file browser",
    )

    args = parser.parse_args(args)

    if args.list:
        print(cache_dir)

    if args.open:
        logging.info("Open cache directory {}".format(cache_dir))
        dispatch_open(cache_dir)
