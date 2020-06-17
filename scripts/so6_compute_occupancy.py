import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from shapely.ops import cascaded_union
from tqdm import tqdm

from traffic.core.time import to_datetime
from traffic.data import SO6, nm_airspaces


def occupancy(data, configuration):
    return len(data.intersects(configuration))


def compute_stats(
    input_file: Path,
    output_file: Optional[Path],
    sector_list: List[str],
    max_workers: int,
    interval: int,
    starting_from: Optional[str],
    ending_at: Optional[str],
) -> pd.DataFrame:

    so6 = SO6.from_file(input_file.as_posix())
    if so6 is None:
        raise RuntimeError

    total: List[Dict[str, int]] = []

    if starting_from is None:
        start_time = so6.data.time1.min()
    else:
        start_time = max(to_datetime(starting_from), so6.data.time1.min())
    if ending_at is None:
        end_time = so6.data.time2.max()
    else:
        end_time = min(to_datetime(ending_at), so6.data.time2.max())

    if end_time < start_time:
        msg = f"End time {end_time} is anterior to start time {start_time}"
        raise ValueError(msg)

    # First clip
    so6 = so6.between(start_time, end_time)

    delta = timedelta(minutes=interval)
    size_range = int((end_time - start_time) / delta) + 1
    time_list = [start_time + i * delta for i in range(size_range)]

    all_sectors = [nm_airspaces[airspace] for airspace in sector_list]
    so6 = so6.inside_bbox(
        cascaded_union([s.flatten() for s in all_sectors if s is not None])
    )

    for start_ in tqdm(time_list):
        subset = so6.between(start_, delta)
        args = {}
        # subset serializes well as it is much smaller than so6
        # => no multiprocess on so6!!
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = {
                executor.submit(occupancy, subset, sector): sector.name
                for sector in all_sectors
                if sector is not None
            }
            for future in as_completed(tasks):
                conf = tasks[future]
                try:
                    args[conf] = future.result()
                except Exception as e:
                    print(f"Exception {e} raised on {conf}")
        total.append(args)

    stats = pd.DataFrame.from_dict(total)
    stats.index = time_list

    if output_file is not None:
        if output_file.suffix == ".pkl":
            stats.to_pickle(output_file.as_posix())
        elif output_file.suffix == ".csv":
            stats.to_csv(output_file.as_posix())
        elif output_file.suffix == ".xlsx":
            stats.to_excel(output_file.as_posix())
        else:
            print(stats)

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Statistics of occupancy on a SO6 file"
    )

    parser.add_argument(
        "-i",
        dest="interval",
        default=10,
        type=int,
        help="number of minutes for a time window",
    )
    parser.add_argument(
        "-o", dest="output_file", type=Path, help="output file for results"
    )
    parser.add_argument(
        "-t",
        dest="max_workers",
        default=4,
        type=int,
        help="number of parallel processes",
    )
    parser.add_argument(
        "-f", dest="starting_from", help="start time (yyyy:mm:ddThh:mm:ssZ)"
    )
    parser.add_argument(
        "-u", dest="ending_at", help="end time (yyyy:mm:ddThh:mm:ssZ)"
    )

    parser.add_argument("input_file", type=Path, help="SO6 file to parse")

    parser.add_argument(
        "sector_list",
        nargs="+",
        help="list of airspaces to pick in AIRAC files",
    )

    args = parser.parse_args()
    res = compute_stats(**vars(args))
