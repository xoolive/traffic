import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from traffic.core import Airspace
from traffic.data import SO6, nm_airspaces


def clip(so6: SO6, airspace: Airspace) -> SO6:
    return so6.inside_bbox(airspace).intersects(airspace)


def unpack_and_clip(filename: str, sectorname: str) -> SO6:
    so6 = SO6.from_file(filename)
    sector = nm_airspaces[sectorname]
    if sector is None:
        raise ValueError("Airspace not found")
    if so6 is None:
        raise RuntimeError
    return clip(so6, sector)


def prepare_all(filename: Path, output_dir: Path, sectorname: str) -> None:
    so6 = unpack_and_clip(filename.as_posix(), sectorname)
    output_name = filename.with_suffix(".pkl").name
    so6.to_pickle((output_dir / output_name).as_posix())


def glob_all(
    directory: Path, output_dir: Path, sectorname: str, max_workers: int = 4
) -> None:

    if not directory.is_dir():
        raise ValueError(f"Directory {directory} does not exist")

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = {
            executor.submit(
                prepare_all, filename, output_dir, sectorname
            ): filename
            for filename in directory.glob("**/*.so6")
        }
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            try:
                future.result()
            except Exception as e:
                print(f"Exception {e} occurred on file {tasks[future]}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clip SO6 on sector")

    parser.add_argument(
        "-d", dest="directory", type=Path, help="directory containing so6 files"
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=Path,
        help="output directory for pkl files",
    )
    parser.add_argument(
        "-s",
        dest="sector_name",
        help="name of the sector to pick in AIRAC files",
    )
    parser.add_argument(
        "-t",
        dest="max_workers",
        default=4,
        type=int,
        help="number of parallel processes",
    )

    args = parser.parse_args()

    glob_all(**vars(args))
