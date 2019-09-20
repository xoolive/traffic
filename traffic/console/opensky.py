import argparse
import logging
from pathlib import Path


def opensky_data(start, stop, output_file, **kwargs):

    from ..data import nm_airspaces, opensky
    from ..drawing import location

    if kwargs["bounds"] is not None:
        bounds = kwargs["bounds"]
        if "," in bounds:
            kwargs["bounds"] = tuple(float(f) for f in bounds.split(","))
        else:
            try:
                sector = nm_airspaces[bounds]
                if sector is not None:
                    kwargs["bounds"] = sector
                else:
                    raise Exception
            except Exception:
                # ask OpenStreetMap
                kwargs["bounds"] = location(bounds)

    if "verbose" in kwargs:
        del kwargs["verbose"]

    data = opensky.history(start, stop, **kwargs)
    assert data is not None

    if output_file.suffix == ".pkl":
        data.to_pickle(output_file.as_posix())

    if output_file.suffix == ".csv":
        data.to_csv(output_file.as_posix())

    if output_file.suffix == ".h5":
        data.to_hdf(output_file.as_posix())

    if output_file.suffix == ".json":
        data.to_json(output_file.as_posix())

    if output_file.suffix == ".xlsx":
        data.to_excel(output_file.as_posix())


def main(args):

    parser = argparse.ArgumentParser(
        prog="traffic opensky",
        description="Get data from OpenSky Impala records",
    )

    parser.add_argument("start", help="start date for records")
    parser.add_argument(
        "-s", dest="stop", default=None, help="end date for records"
    )
    parser.add_argument(
        "-o", dest="output_file", type=Path, help="output file for records"
    )
    parser.add_argument("-c", dest="callsign", help="callsign for flight")
    parser.add_argument(
        "-b",
        dest="bounds",
        default=None,
        help="bounding box for records (sector name or WSEN)",
    )

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

    opensky_data(**vars(args))
