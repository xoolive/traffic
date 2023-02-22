import argparse
import logging
from typing import List

from rich.console import Console


def main(args_list: List[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="traffic data",
        description="Explore basic navigational data embedded with traffic",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--aircraft",
        "-a",
        dest="aircraft",
        action="store_true",
        help="followed by registration or transponder identification",
    )
    group.add_argument(
        "--airport",
        "-p",
        dest="airport",
        action="store_true",
        help="followed by IATA, ICAO codes, country or city names",
    )
    group.add_argument(
        "--navaid",
        "-n",
        dest="navaid",
        action="store_true",
        help="followed by navigational beacon name",
    )
    group.add_argument(
        "--operator",
        "-o",
        dest="operator",
        action="store_true",
        help="followed by the name of the operator (list all aircraft)",
    )
    group.add_argument(
        "--stats",
        "-s",
        dest="stats",
        action="store_true",
        help="followed by the name of the operator (stats of all aircraft)",
    )

    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="display logging messages",
    )

    parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(args_list)

    logger = logging.getLogger()
    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    from ..data import aircraft, airports, navaids

    console = Console()

    if args.aircraft:
        for arg in args.args:
            console.print(aircraft[arg])

    if args.operator:
        console.print(aircraft.operator(" ".join(args.args)))

    if args.stats:
        console.print(aircraft.stats(" ".join(args.args)))

    if args.airport:
        subset = airports
        for arg in args.args:
            subset = subset.search(arg)
        console.print(subset)

    if args.navaid:
        for arg in args.args:
            for navaid in navaids.search(arg):
                console.print(navaid)
