from typing import Tuple, cast
from cartopy.crs import EuroPP  # type: ignore
import numpy as np

from traffic.core import Traffic
from traffic.data.samples import collections, get_sample

class StupidGeneration:
    """ Special Generation just for a test without sklearn or PyTorch.
    Generation model generates the first flight it saw.
    """

    def fit(self, X: np.ndarray, **kwargs) -> "StupidGeneration":
        self.x = X[0]

        return self

    def samples(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.repeat(self.x[np.newaxis, ...], n_samples, axis=0), np.array([])
        )


def test_generation() -> None:
    switzerland: Traffic = get_sample(collections, "switzerland")

    smaller = cast(
        Traffic,
        switzerland.between("2018-08-01 12:00", "2018-08-01 14:00")
        .assign_id()
        .resample(10)
        .eval(max_workers=4)
    )

    smaller = smaller.compute_xy(projection=EuroPP())

    smaller = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in smaller
    )

    g = smaller.generation(
        generation=StupidGeneration(),
        features=["track", "groundspeed", "altitude", "timedelta"],
    )

    t_gen = Traffic(g.sample(4))

    assert 