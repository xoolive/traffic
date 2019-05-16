from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import pyproj
from cartopy import crs
from typing_extensions import Protocol

if TYPE_CHECKING:
    from ..core import Traffic  # noqa: F401


class Transformer(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...


class Clustering(Protocol):
    def fit(self, X: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


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

    if "last_position" in traffic.data.columns:
        traffic = traffic.drop(columns="last_position")

    traffic = traffic.resample(nb_samples).eval(max_workers=max_workers)

    if all(
        [
            "x" in features,
            "y" in features,
            "x" not in traffic.data.columns,
            "y" not in traffic.data.columns,
        ]
    ):
        if projection is None:
            raise RuntimeError(
                "No 'x' and 'y' columns nor projection method passed"
            )
        traffic = traffic.compute_xy(projection)

    X = np.stack(list(f.data[features].values.ravel() for f in traffic))

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
