import argparse
import logging
import os
import subprocess
import sys

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
    group.add_argument(
        "--stats",
        "-s",
        dest="stats",
        action="store_true",
        help="print extended stats about each Flight in the traffic file",
    )

    args = parser.parse_args(args)
    t = Traffic.from_file(args.filename)

    print("Traffic with {} identifiers".format(len(t)))
    print(t)
    print()
    print("with the following features:")
    print(t.data.columns)

    if args.head:
        print()
        print("Head of the DataFrame:")
        print(t.data.head())

    elif args.stats:
        raise NotImplementedError
