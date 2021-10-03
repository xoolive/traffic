from typing import Tuple, cast

import pytest

import numpy as np
from cartopy.crs import EuroPP
from traffic.core import Flight, Traffic
from traffic.data.samples import collections, get_sample


class StupidGeneration:
    """Special Generation just for a test without sklearn or PyTorch.
    Generation model generates the first flight it saw.
    """

    def fit(self, X: np.ndarray, **kwargs) -> "StupidGeneration":
        self.x = X[0]

        return self

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.repeat(self.x[np.newaxis, ...], n_samples, axis=0),
            np.array([]),
        )


def test_generation() -> None:
    switzerland: Traffic = get_sample(collections, "switzerland")

    smaller = cast(
        Traffic,
        switzerland.between("2018-08-01 12:00", "2018-08-01 14:00")
        .assign_id()
        .resample(10)
        .eval(max_workers=4),
    )

    smaller = smaller.compute_xy(projection=EuroPP())

    t = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in smaller
    )

    assert isinstance(t, Traffic)

    g = t.generation(
        generation=StupidGeneration(),
        features=["track", "groundspeed", "altitude", "timedelta"],
    )
    df = g.sample(5, coordinates={"latitude": 15, "longitude": 15})
    t_gen = Traffic(df)

    assert isinstance(t_gen, Traffic)
    assert len(t_gen) == 5
    assert isinstance(t_gen[0], Flight)

    g = t.generation(
        generation=StupidGeneration(),
        features=["x", "y", "altitude", "timedelta"],
    )
    df = g.sample(6, projection=EuroPP())
    t_gen = Traffic(df)

    assert isinstance(t_gen, Traffic)
    assert len(t_gen) == 6
    assert isinstance(t_gen[0], Flight)

    g = t.generation(
        generation=StupidGeneration(),
        features=["altitude", "timedelta"],
    )

    with pytest.warns(UserWarning):
        g.sample(2)
