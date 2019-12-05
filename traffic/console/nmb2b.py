import argparse
import logging

description = """
Get data from Network Manager B2B Service.
"""


def main(args):

    parser = argparse.ArgumentParser(
        prog="traffic nmb2b", description=description
    )

    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="display logging messages",
    )

    parser.add_argument(
        "-a", "--airac", dest="airac", default=None, help="AIRAC version"
    )

    args = parser.parse_args(args)

    logger = logging.getLogger()
    if args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    if args.airac is not None:
        from traffic.data import nm_b2b

        nm_b2b.aixm_dataset(args.airac)
    else:
        raise RuntimeError("No action requested")
