import argparse
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

description = """
Get data from Radarcape or dump1090. Timestamp each message and dump it to a
file. The reference parameter is first taken for the [decoders] section in the
configuration file. If no such parameters exists, then we search for a dump1090
output on localhost:30005 and set the reference airport as reference.
"""


def main(args):

    parser = argparse.ArgumentParser(
        prog="traffic dump", description=description
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        default="output.pkl",
        help="destination file (default: output.pkl)",
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="display logging messages",
    )
    parser.add_argument(
        "reference", help="configuration name or IATA/ICAO code for dump1090"
    )

    args = parser.parse_args(args)

    logger = logging.getLogger()
    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    from traffic import config
    from traffic.data import ModeS_Decoder

    now = datetime.now(timezone.utc)
    filename = Path(now.strftime(args.output.as_posix()))

    try:
        address = config.get("decoders", args.reference)
        host_port, reference = address.split("/")
        host, port = host_port.split(":")
        decoder = ModeS_Decoder.from_address(
            host, int(port), reference, filename.with_suffix(".csv").as_posix()
        )
    except Exception:
        logging.info("fallback to dump1090")
        decoder = ModeS_Decoder.from_dump1090(
            args.reference, filename.with_suffix(".csv").as_posix()
        )

    def signal_handler(sig, frame):
        logging.info("Interruption signal caught")
        t = decoder.traffic
        if t is not None:
            pkl_file = filename.with_suffix(".pkl")
            t.to_pickle(os.path.expanduser(pkl_file))
            logging.info(f"Traffic saved to {pkl_file}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    logging.info("Press Ctrl+C to quit")

    while True:
        continue
