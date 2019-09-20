import argparse
import logging
from pathlib import Path

from ..data.adsb.decode import Decoder


def main(args):
    parser = argparse.ArgumentParser(
        prog="traffic decode",
        description="Decode ADS-B and EHS messages from file",
    )

    parser.add_argument("file", help="path to the file to decode", type=Path)

    parser.add_argument(
        "reference", help="reference airport for decoding surface position"
    )

    parser.add_argument("-o", "--output", help="output pickle file", type=Path)

    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="display logging messages",
    )

    args = parser.parse_args(args)

    logger = logging.getLogger()
    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    decoder = Decoder.from_file(args.file, args.reference)
    assert decoder.traffic is not None
    decoder.traffic.to_pickle(
        args.output
        if args.output is not None
        else args.file.with_suffix(".pkl")
    )
