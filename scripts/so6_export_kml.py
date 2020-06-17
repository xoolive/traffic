import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from traffic.data import SO6, nm_airspaces
from traffic.drawing import kml


def so6_to_kml(
    input_file: Path,
    output_file: Path,
    sector_name: Optional[str],
    start_time: Optional[str],
    stop_time: Optional[str],
) -> None:

    so6 = SO6.from_file(input_file.as_posix())
    if so6 is None:
        raise RuntimeError

    if start_time is not None and stop_time is not None:
        so6 = so6.between(start_time, stop_time)

    if sector_name is not None:
        sector = nm_airspaces[sector_name]
        if sector is None:
            raise ValueError(f"Unknown airspace {sector_name}")
        so6 = so6.inside_bbox(sector).intersects(sector)

    with kml.export(output_file.as_posix()) as doc:
        if sector is not None:
            doc.append(
                sector.export_kml(color="blue", alpha=0.3)  # type: ignore
            )

        # iterate on so6 yield callsign, flight
        for callsign, flight in tqdm(so6):
            doc.append(flight.export_kml(color="#aa3a3a"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export SO6 to KML")

    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default="output.kml",
        help="outfile file (.kml)",
    )

    parser.add_argument(
        "-s",
        dest="sector_name",
        help="name of the sector to pick in AIRAC files",
    )

    parser.add_argument(
        "-f", dest="start_time", help="start_time to filter trajectories"
    )

    parser.add_argument(
        "-t", dest="stop_time", help="stop_time to filter trajectories"
    )

    parser.add_argument("input_file", type=Path, help="so6 file to parse")

    args = parser.parse_args()
    so6_to_kml(**vars(args))
