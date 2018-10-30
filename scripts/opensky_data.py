from pathlib import Path

from traffic.data import airac as sectors
from traffic.data import opensky
from traffic.core.logging import loglevel


def opensky_data(start, stop, output_file, **kwargs):

    if kwargs["bounds"] is not None:
        bounds = kwargs["bounds"]
        if "," in bounds:
            kwargs["bounds"] = tuple(float(f) for f in bounds.split(","))
        else:
            kwargs["bounds"] = sectors[bounds]

    if 'verbose' in kwargs:
        del kwargs['verbose']

    data = opensky.history(start, stop, **kwargs)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get data from OpenSky records"
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

    parser.add_argument("-v", dest="verbose", action="count", default=0,
                        help="display logging messages")

    args = parser.parse_args()

    if args.verbose == 1:
        loglevel('INFO')
    elif args.verbose >= 2:
        loglevel('DEBUG')

    opensky_data(**vars(args))
