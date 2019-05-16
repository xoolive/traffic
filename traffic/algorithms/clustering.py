from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
import pyproj
from cartopy import crs

if TYPE_CHECKING:
    from ..core import Traffic  # noqa: F401


def clustering(
    traffic: "Traffic",
    method,  # must implement fit and predict
    nb_samples: int,
    projection: Union[crs.Projection, pyproj.Proj],
    altitude: Optional[str] = None,
    max_workers: int = 1,
    return_traffic: bool = True,
    transform=None,  # must implement fit_transform
    **kwargs,
) -> "Traffic":

    if "last_position" in traffic.data.columns:
        traffic = traffic.drop(columns="last_position")

    traffic = traffic.resample(nb_samples).eval(max_workers=max_workers)
    columns = ["x", "y"]

    if altitude is not None:
        columns.append(altitude)

    X = np.stack(
        list(
            f.compute_xy(projection).data[columns].values.ravel()
            for f in traffic
        )
    )

    if transform is not None:
        X = transform.fit_transform(X)

    clustering = method(**kwargs)
    clustering.fit(X)

    labels = (
        clustering.labels_
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
