from typing import cast

from cartes.crs import CH1903p  # type: ignore

from traffic.core import Traffic
from traffic.data.samples import collections, get_sample


class StupidClustering:
    """Special Clustering just for a test without sklearn
    Dumb clustering assigning first flight to cluster 0,
    second to 1, etc.
    """

    def fit(self, X):
        self.labels_ = [i % 2 for i, _ in enumerate(X.T)]

    def predict(self, X):
        pass


def test_clustering() -> None:
    switzerland: Traffic = get_sample(collections, "switzerland")

    smaller = cast(
        Traffic,
        switzerland.between("2018-08-01 12:00", "2018-08-01 14:00")
        .assign_id()
        .eval(max_workers=4),
    )

    t_clustering = smaller.clustering(
        nb_samples=15,
        projection=CH1903p(),
        features=["x", "y"],
        clustering=StupidClustering(),
    ).fit_predict()

    v1, v2 = (
        t_clustering.groupby(["cluster"])
        .agg({"flight_id": "nunique"})
        .flight_id
    )

    assert abs(v1 - v2) <= 1
