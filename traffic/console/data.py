import argparse


def main(args):

    from ..data import airports, aircraft, navaids

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

    parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(args)

    if args.aircraft:
        for arg in args.args:
            print(aircraft[arg])

    if args.operator:
        print(aircraft.operator(" ".join(args.args)))

    if args.stats:
        print(aircraft.stats(" ".join(args.args)))

    if args.airport:
        for arg in args.args:
            for airport in airports.search(arg):
                print(airport)

    if args.navaid:
        for arg in args.args:
            for navaid in navaids.search(arg):
                print(navaid)
