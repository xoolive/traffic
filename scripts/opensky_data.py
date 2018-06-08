from pathlib import Path

from tqdm import tqdm
from traffic.data import opensky, sectors


def opensky_data(date, after, output_file, **kwargs):

    if kwargs['bounds'] is not None:
        kwargs['bounds'] = sectors[kwargs['bounds']]

    data = opensky.history(date, after, progressbar=tqdm, **kwargs)

    if output_file.suffix == '.pkl':
        data.to_pickle(output_file.as_posix())

    if output_file.suffix == '.csv':
        data.to_csv(output_file.as_posix())

    if output_file.suffix == '.h5':
        data.to_hdf(output_file.as_posix())

    if output_file.suffix == '.json':
        data.to_json(output_file.as_posix())

    if output_file.suffix == '.xlsx':
        data.to_excel(output_file.as_posix())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Get data from OpenSky records")

    parser.add_argument("date", help="start date for records")
    parser.add_argument("-a", dest="after", default=None,
                        help="end date for records")
    parser.add_argument("-o", dest="output_file", type=Path,
                        help="output file for records")
    parser.add_argument("-c", dest="callsign",
                        help="callsign for flight")
    parser.add_argument("-b", dest="bounds", default=None,
                        help="bounding box for records (sector name)")

    args = parser.parse_args()

    opensky_data(**vars(args))
