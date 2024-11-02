from typing import Any, Tuple, cast

from cartopy.crs import EuroPP

import numpy as np
import numpy.typing as npt
import pandas as pd
from traffic.core import Flight, Traffic
from traffic.data.samples import collections, get_sample


class NaiveGeneration:
    """Special Generation just for a test without sklearn or PyTorch.
    Generation model generates the first flight it saw.
    """

    def fit(
        self, X: npt.NDArray[np.float64], **kwargs: Any
    ) -> "NaiveGeneration":
        self.x = X[0]

        return self

    def sample(
        self, n_samples: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return (
            np.repeat(self.x[np.newaxis, ...], n_samples, axis=0),
            np.array([]),
        )


def test_generation() -> None:
    switzerland = cast(Traffic, get_sample(collections, "switzerland"))

    def compute_timedelta(df: pd.DataFrame) -> pd.Series:
        return (df.timestamp - df.timestamp.min()).dt.total_seconds()

    between = switzerland.between("2018-08-01 12:00", "2018-08-01 14:00")
    assert between is not None
    smaller = (
        between.assign_id()
        .resample(10)
        .compute_xy(projection=EuroPP())
        .assign(timedelta=compute_timedelta)
        .eval()
    )

    assert isinstance(smaller, Traffic)

    g = smaller.generation(
        generation=NaiveGeneration(),
        features=["track", "groundspeed", "altitude", "timedelta"],
    )
    t_gen = g.sample(5, coordinates={"latitude": 15, "longitude": 15})

    assert isinstance(t_gen, Traffic)
    assert len(t_gen) == 5
    assert isinstance(t_gen[0], Flight)

    g = smaller.generation(
        generation=NaiveGeneration(),
        features=["x", "y", "altitude", "timedelta"],
    )
    t_gen = g.sample(6, projection=EuroPP())

    assert isinstance(t_gen, Traffic)
    assert len(t_gen) == 6
    assert isinstance(t_gen[0], Flight)
