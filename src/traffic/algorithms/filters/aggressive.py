from typing import ClassVar, TypedDict

from typing_extensions import NotRequired

import numpy as np
import pandas as pd

from . import FilterBase


class DerivativeParams(TypedDict):
    first: float  # threshold for 1st derivative
    second: float  # threshold for 2nd derivative
    kernel: int


class FilterDerivative(FilterBase):
    """Filter based on the 1st and 2nd derivatives of parameters

    The method computes the absolute value of the 1st and 2nd derivatives
    of the parameters. If the value of the derivatives is above the defined
    threshold values, the datapoint is removed

    """

    # default parameter values
    default: ClassVar[dict[str, DerivativeParams]] = dict(
        altitude=dict(first=200, second=150, kernel=10),
        geoaltitude=dict(first=200, second=150, kernel=10),
        vertical_rate=dict(first=1500, second=1000, kernel=5),
        groundspeed=dict(first=12, second=10, kernel=3),
        track=dict(first=12, second=10, kernel=2),
    )

    def __init__(
        self, time_column: str = "timestamp", **kwargs: DerivativeParams
    ) -> None:
        """

        :param time_column: the name of the time column (default: "timestamp")

        :param kwargs: each keyword argument has the name of a feature.
            the value must be a dictionary with the following keys:
            - first: threshold value for the first derivative
            - second: threshold value for the second derivative
            - kernel: the kernel size in seconds

        If two spikes are detected within the width of the kernel, all
        datapoints inbetween are also removed.

        """
        self.columns = {**self.default, **kwargs}
        self.time_column = time_column

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        timediff = data[self.time_column].diff().dt.total_seconds()
        for column, params in self.columns.items():
            if column not in data.columns:
                continue
            window = params["kernel"]
            diff1 = data[column].diff().abs()
            # TODO it could be smarter to use the unwrapped version...
            if column == "track":
                diff1.loc[(diff1 < 370) & (diff1 > 350)] = 0
            diff2 = diff1.diff().abs()

            deriv1 = diff1 / timediff
            deriv2 = diff2 / timediff
            spike = np.bitwise_or(
                (deriv1 >= params["first"]), (deriv2 >= params["second"])
            )
            spike = spike.fillna(False, inplace=False)
            spike_time = pd.Series(
                pd.Timestamp("NaT"), index=data.index
            ).convert_dtypes(dtype_backend="pyarrow")
            spike_time = spike_time.dt.tz_localize("UTC").copy()
            spike_time.loc[spike] = data.loc[spike, self.time_column]

            if not spike_time.isnull().all():
                spike_time_prev = spike_time.ffill()
                spike_delta_prev = data["timestamp"] - spike_time_prev
                spike_time_next = spike_time.bfill()
                spike_delta_next = spike_time_next - data["timestamp"]
                in_window = np.bitwise_and(
                    spike_delta_prev.dt.total_seconds() <= window,
                    spike_delta_next.dt.total_seconds() <= window,
                )
                data.loc[in_window, column] = np.nan

        return data


class ClusteringParams(TypedDict):
    group_size: int
    value_threshold: float
    time_threshold: NotRequired[float]


class FilterClustering(FilterBase):
    """Filter based on clustering.

    The method creates clusters of datapoints based on the difference in time
    and parameter value. If the cluster is larger than the defined group size
    the datapoints are kept, otherwise they are removed.

    """

    default: ClassVar[dict[str, ClusteringParams]] = dict(
        altitude=dict(group_size=15, value_threshold=500),
        geoaltitude=dict(group_size=15, value_threshold=500),
        vertical_rate=dict(group_size=15, value_threshold=500),
        onground=dict(group_size=15, value_threshold=500),
        track=dict(group_size=15, value_threshold=500),
        latitude=dict(group_size=15, value_threshold=500),
        longitude=dict(group_size=15, value_threshold=500),
    )

    def __init__(
        self, time_column: str = "timestamp", **kwargs: ClusteringParams
    ) -> None:
        """
        :param time_column: the name of the time column (default: "timestamp")

        :param kwargs: each keyword argument has the name of a feature.
            the value must be a dictionary with the following keys:
            - group_size: minimum size of the cluster to be kept
            - value_threshold: within the value threshold, the samples fall in
              the same cluster
            - time_threshold: within the time threshold, the samples fall in
              the same cluster
        """
        self.columns = {**self.default, **kwargs}
        self.time_column = time_column

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop_duplicates([self.time_column], keep="last")

        for column, param in self.columns.items():
            if column not in data.columns:
                continue
            if column == "onground":
                statechange = data[column].diff().astype(bool)
                data["group"] = statechange.eq(True).cumsum()
                groups = data["group"].value_counts()
                keepers = groups[groups > param["group_size"]].index.tolist()
                data[column] = data[column].where(
                    data["group"].isin(keepers), float("NaN")
                )

            else:
                timediff = data[self.time_column].diff().dt.total_seconds()
                temp_index = data.index
                temp_values = data[column].dropna()
                paradiff = temp_values.diff().reindex(temp_index).abs()
                bigdiff = pd.Series(
                    np.bitwise_or(
                        (timediff > param.get("time_threshold", 60)),
                        (paradiff > param["value_threshold"]),
                    )
                )
                data["group"] = bigdiff.fillna(False).astype(int).cumsum()
                groups = data[data[column].notna()]["group"].value_counts()
                keepers = groups[groups > param["group_size"]].index.tolist()
                data[column] = data[column].where(
                    data["group"].isin(keepers), float("NaN")
                )

        return data.drop(columns=["group"])
