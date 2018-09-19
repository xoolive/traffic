from collections import UserDict, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import Callable, Dict, Iterator, NamedTuple, Optional

import numpy as np

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm.autonotebook import tqdm

from ..core import Traffic
from ..core.time import round_time


class Distance(NamedTuple):

    timestamp: datetime
    vertical: float
    lateral: float
    aggregated: float
    index1: int
    index2: int
    reg1: str
    reg2: str
    lat1: float
    lon1: float
    lat2: float
    lon2: float


class CPADict(UserDict):
    def __missing__(self, key):
        p1, p2 = key
        if (p2, p1) in self.keys():
            return self[p2, p1]
        raise KeyError

    def __setitem__(self, key, value: Distance) -> None:
        current: Optional[Distance] = self.get(key, None)
        if current is not None and value.aggregated > current.aggregated:
            return None
        UserDict.__setitem__(self, key, value)


def compute_cpa(
    traffic: Traffic,
    rounding_time: timedelta = timedelta(minutes=30),
    lateral_sep: float = 5 * 1852,  # 5 nautical miles to meters
    vertical_sep: float = 1000,  # feet
    threshold: float = 3,
    progressbar: Callable[..., Iterator] = tqdm,
) -> pd.DataFrame:

    cpadict: Dict[datetime, CPADict] = defaultdict(CPADict)
    total = (traffic.end_time - traffic.start_time).seconds + 1
    altitude = (
        "baro_altitude"
        if "baro_altitude" in traffic.data.columns
        else "altitude"
    )

    if "x" not in traffic.data.columns or "y" not in traffic.data.columns:
        traffic = traffic.compute_xy()

    # TODO bugfix
    traffic = Traffic(traffic.data.reset_index())
        
    for ts, d in progressbar(traffic.groupby("timestamp"), total=total):

        cpa = cpadict[round_time(ts, by=rounding_time)]

        z = d[altitude].values
        z = z.reshape((len(z), 1))

        lateral_dist = pdist(np.c_[d.x, d.y])  # TODO euclidean/wgs84
        vertical_dist = squareform(np.abs(z - z.transpose()))

        for ((i1, i2), vertical, lateral, aggregated) in zip(
            combinations(d.index, 2),
            vertical_dist,
            lateral_dist,
            np.maximum(
                lateral_dist / lateral_sep, vertical_dist / vertical_sep
            ),
        ):

            if aggregated > threshold:
                continue

            x1, x2 = d.loc[i1], d.loc[i2]
            cpa[x1.icao24, x2.icao24] = Distance(
                ts,
                vertical,
                lateral,
                aggregated,
                i1,
                i2,
                x1.icao24,
                x2.icao24,
                x1.latitude,
                x1.longitude,
                x2.latitude,
                x2.longitude,
            )

    return pd.concat(
        [
            pd.DataFrame.from_records(
                [x._asdict() for x in cpa.values()]
            ).assign(timeslot=key)
            for key, cpa in cpadict.items()
        ]
    )
