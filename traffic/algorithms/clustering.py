from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import pyproj
from cartopy import crs
from scipy.spatial.distance import pdist, squareform
from typing_extensions import Protocol

if TYPE_CHECKING:
    from ..core import Flight, Traffic  # noqa: F401


class Transformer(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...


class Clustering(Protocol):
    def fit(self, X: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def prepare_features(
    traffic: "Traffic",
    nb_samples: int,
    features: List[str] = ["x", "y"],
    projection: Union[None, crs.Projection, pyproj.Proj] = None,
    max_workers: int = 1,
) -> np.ndarray:
    if "last_position" in traffic.data.columns:
        traffic = traffic.drop(columns="last_position")

    resampled = traffic.resample(nb_samples).eval(max_workers=max_workers)

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


def clustering(
    traffic: "Traffic",
    clustering: Clustering,
    nb_samples: int,
    features: List[str] = ["x", "y"],
    *args,
    projection: Union[None, crs.Projection, pyproj.Proj] = None,
    transform: Optional[Transformer] = None,
    max_workers: int = 1,
    return_traffic: bool = True,
) -> "Traffic":

    X = prepare_features(traffic, nb_samples, features, projection, max_workers)

    if transform is not None:
        X = transform.fit_transform(X)

    clustering.fit(X)

    labels: np.ndarray = (
        clustering.labels_  # type: ignore
        if hasattr(clustering, "labels_")
        else clustering.predict(X)
    )

    clusters = pd.DataFrame.from_records(
        [
            dict(flight_id=f.flight_id, cluster=cluster_id)
            for f, cluster_id in zip(traffic, labels)
        ]
    )
    if not return_traffic:
        return clusters

    return traffic.merge(clusters, on="flight_id")


def centroid(
    traffic: "Traffic",
    nb_samples: int,
    features: List[str] = ["x", "y"],
    projection: Union[None, crs.Projection, pyproj.Proj] = None,
    transform: Optional[Transformer] = None,
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
