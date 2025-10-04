import argparse
import logging
from typing import List

from . import dispatch_open

logger = logging.getLogger(__name__)


def main(args_list: List[str]) -> None:
    from .. import cache_path

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
    group.add_argument(
        "--fill",
        "-f",
        dest="fill",
        action="store_true",
        help="download necessary data to put in cache before tests",
    )

    args = parser.parse_args(args_list)

    if args.list:
        print(cache_path)

    if args.open:
        logger.info("Open cache directory {}".format(cache_path))
        dispatch_open(cache_path)

    if args.fill:
        from traffic.data import aircraft, airports, navaids
        from traffic.data.datasets import landing_zurich_2019
        from traffic.data.datasets.scat import SCAT

        p = airports["EHAM"]
        n = navaids["NARAK"]
        a = aircraft["F-HNAV"]
        assert p is not None and n is not None and a is not None

        s = SCAT("scat20161015_20161021.zip", nflights=10)
        assert s is not None

        f = landing_zurich_2019[0]
        assert f is not None
