from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import pyproj
from cartopy import crs
from scipy.spatial.distance import pdist, squareform
from typing_extensions import Protocol

if TYPE_CHECKING:
    from ..core import Flight, Traffic  # noqa: F401


class TransformerProtocol(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


class ClusteringProtocol(Protocol):
    def fit(self, X: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def prepare_features(
    traffic: "Traffic",
    nb_samples: Optional[int],
    features: List[str],
    projection: Union[None, crs.Projection, pyproj.Proj] = None,
    max_workers: int = 1,
) -> np.ndarray:
    if "last_position" in traffic.data.columns:
        traffic = traffic.drop(columns="last_position")

    resampled = traffic
    if nb_samples is not None:
        _resampled = traffic.resample(nb_samples).eval(max_workers=max_workers)
        # TODO LazyTraffic/LazyOptionalTraffic
        assert _resampled is not None
        resampled = _resampled

    if all(
        [
            "x" in features,
            "y" in features,
            "x" not in resampled.data.columns,
            "y" not in resampled.data.columns,
        ]
    ):
        if projection is None:
            raise RuntimeError(
                "No 'x' and 'y' columns nor projection method passed"
            )
        resampled = resampled.compute_xy(projection)

    return np.stack(list(f.data[features].values.ravel() for f in resampled))


class Clustering:
    def __init__(
        self,
        traffic: "Traffic",
        clustering: ClusteringProtocol,
        nb_samples: Optional[int],
        features: List[str],
        *args,
        projection: Union[None, crs.Projection, pyproj.Proj] = None,
        transform: Optional[TransformerProtocol] = None,
    ) -> None:

        self.traffic = traffic
        self.clustering = clustering
        self.nb_samples = nb_samples
        self.features = features
        self.projection = projection
        self.transform = transform
        self.X: Optional[np.ndarray] = None

    def fit(self, max_workers: int = 1) -> None:

        if self.X is None:
            self.X = prepare_features(
                self.traffic,
                self.nb_samples,
                self.features,
                self.projection,
                max_workers,
            )

            if self.transform is not None:
                self.X = self.transform.fit_transform(self.X)

        self.clustering.fit(self.X)

    def predict(self, max_workers: int = 1, return_traffic: bool = True):

        if self.X is None:
            self.X = prepare_features(
                self.traffic,
                self.nb_samples,
                self.features,
                self.projection,
                max_workers,
            )

            if self.transform is not None:
                self.X = self.transform.transform(self.X)

        labels = self.clustering.predict(self.X)

        clusters = pd.DataFrame.from_records(
            [
                dict(flight_id=f.flight_id, cluster=cluster_id)
                for f, cluster_id in zip(self.traffic, labels)
            ]
        )
        if not return_traffic:
            return clusters

        return self.traffic.merge(clusters, on="flight_id")

    def fit_predict(
        self, max_workers: int = 1, return_traffic: bool = True
    ) -> "Traffic":

        if self.X is None:
            self.X = prepare_features(
                self.traffic,
                self.nb_samples,
                self.features,
                self.projection,
                max_workers,
            )

            if self.transform is not None:
                self.X = self.transform.fit_transform(self.X)

        self.clustering.fit(self.X)

        labels: np.ndarray = (
            self.clustering.labels_  # type: ignore
            if hasattr(self.clustering, "labels_")
            else self.clustering.predict(self.X)
        )

        clusters = pd.DataFrame.from_records(
            [
                dict(flight_id=f.flight_id, cluster=cluster_id)
                for f, cluster_id in zip(self.traffic, labels)
            ]
        )
        if not return_traffic:
            return clusters

        return self.traffic.merge(clusters, on="flight_id")


def centroid(
    traffic: "Traffic",
    nb_samples: Optional[int],
    features: List[str],
    projection: Union[None, crs.Projection, pyproj.Proj] = None,
    transform: Optional[TransformerProtocol] = None,
    max_workers: int = 1,
    *args,
    **kwargs,
) -> "Flight":
    """
    Returns the trajectory in the Traffic that is the closest to all other
    trajectories.

    .. warning::
        Remember the time and space complexity of this method is **quadratic**.

    """

    X = prepare_features(traffic, nb_samples, features, projection, max_workers)
    ids = list(f.flight_id for f in traffic)

    if transform is not None:
        X = transform.fit_transform(X)

    return traffic[
        ids[squareform(pdist(X, *args, **kwargs)).mean(axis=1).argmin()]
    ]
