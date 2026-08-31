from typing import Any, Tuple, cast

import pytest
from cartopy.crs import EuroPP

import numpy as np
import numpy.typing as npt
import pandas as pd
from traffic.algorithms.generation import Generation
from traffic.core import Flight, Traffic
from traffic.data.samples import collections, get_sample


class StandardScaler:
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]

    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, X: npt.NDArray[np.float64]) -> "StandardScaler":
        self.fit_calls += 1
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1
        return self

    def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray((X - self.mean) / self.std, dtype=np.float64)

    def inverse_transform(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.asarray(X * self.std + self.mean, dtype=np.float64)


class LegacyScaler:
    mean: npt.NDArray[np.float64]

    def fit_transform(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        self.mean = X.mean(axis=0)
        return np.asarray(X - self.mean, dtype=np.float64)

    def inverse_transform(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.asarray(X + self.mean, dtype=np.float64)


class NaiveGeneration:
    """Special Generation just for a test without sklearn or PyTorch.
    Generation model generates the first flight it saw.
    """

    def fit(
        self, X: npt.NDArray[np.float64], **kwargs: Any
    ) -> "NaiveGeneration":
        self.x = X[0]
        self.fit_kwargs = kwargs

        return self

    def sample(
        self, n_samples: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return (
            np.repeat(self.x[np.newaxis, ...], n_samples, axis=0),
            np.array([]),
        )


def test_generation_without_model_prepares_and_rebuilds() -> None:
    switzerland = cast(Traffic, get_sample(collections, "switzerland"))
    smaller = switzerland.assign_id().resample(10).eval()
    assert smaller is not None
    scaler = StandardScaler()
    adapter = Generation(
        generation=None,
        features=["latitude", "longitude"],
        scaler=scaler,
    )

    adapter.fit_preprocessing(smaller)
    values = adapter.transform_features(smaller)
    raw_values = adapter.inverse_transform_features(values)
    rebuilt = adapter.build_traffic(raw_values)

    assert len(rebuilt) == len(smaller)
    np.testing.assert_allclose(
        rebuilt.data[["latitude", "longitude"]].to_numpy(),
        raw_values.reshape(-1, 2),
    )

    compatible_adapter = Generation(
        generation=None,
        features=["latitude", "longitude"],
        scaler=StandardScaler(),
    )
    assert np.allclose(compatible_adapter.prepare_features(smaller), values)

    legacy_adapter = Generation(
        generation=NaiveGeneration(),
        features=["latitude", "longitude"],
        scaler=LegacyScaler(),
    ).fit(smaller)
    assert isinstance(legacy_adapter.sample(), Traffic)

    model = NaiveGeneration()
    Generation(
        generation=model,
        features=["latitude", "longitude"],
    ).fit(smaller, example_option=True)
    assert model.fit_kwargs == {"example_option": True}

    with pytest.raises(RuntimeError, match="no external model to fit"):
        adapter.fit(smaller)
    assert scaler.fit_calls == 1
    with pytest.raises(RuntimeError, match="no external model to fit"):
        adapter.fit_prepared(values)
    with pytest.raises(RuntimeError, match="no external model to sample"):
        adapter.sample()


def test_generation_scaler_can_be_fit_before_model() -> None:
    switzerland = cast(Traffic, get_sample(collections, "switzerland"))
    smaller = switzerland.assign_id().resample(10).eval()
    scaler = StandardScaler()
    model = NaiveGeneration()
    generation = Generation(
        generation=model,
        features=["latitude", "longitude"],
        scaler=scaler,
    )
    training = smaller[:1]
    validation = smaller[1:2]
    generation.fit_preprocessing(training)
    train_values = generation.transform_features(training)
    validation_values = generation.transform_features(validation)
    generation.fit_prepared(train_values)
    sampled = generation.sample()

    assert train_values.shape == validation_values.shape
    assert np.allclose(train_values, 0)
    assert scaler.fit_calls == 1
    np.testing.assert_allclose(
        sampled.data[["latitude", "longitude"]].to_numpy(dtype=float),
        training.data[["latitude", "longitude"]].to_numpy(dtype=float),
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
