import argparse
from pathlib import Path


def main(args):

    from ..core import Traffic

    parser = argparse.ArgumentParser(
        prog="traffic show", description="inspect a traffic file"
    )

    parser.add_argument(
        "filename", type=Path, help="path to the traffic file to inspect"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--head",
        dest="head",
        action="store_true",
        help="print the first lines of the traffic file",
    )
    parser.add_argument(
        "callsigns",
        nargs=argparse.REMAINDER,
        help="display specific information about specific flights",
    )

    args = parser.parse_args(args)
    t = Traffic.from_file(args.filename)
    assert t is not None

    print("Traffic with {} identifiers".format(len(t)))
    print(t)
    print()
    print("with the following features:")
    print(t.data.columns)

    if args.head:
        print()
        print("Head of the DataFrame:")
        print(t.data.head())

    if args.callsigns:
        for callsign in args.callsigns:
            print()
            print(t[callsign])
