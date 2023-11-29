from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,  # for python 3.8 and impunity
    Generic,
    Protocol,
    Type,
    TypedDict,
    TypeVar,
    cast,
)

from impunity import impunity
from scipy import signal
from typing_extensions import Annotated, NotRequired

import numpy as np
import numpy.typing as npt
import pandas as pd
import operator
import warnings

NM2METER = 1852


class Filter(Protocol):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


class FilterBase(Filter):
    tracked_variables: dict[str, list[Any]]

    def __or__(self, other: FilterBase) -> FilterBase:
        """Composition operator on filters."""

        def composition(
            object: Type[FilterBase], data: pd.DataFrame
        ) -> pd.DataFrame:
            return other.apply(self.apply(data))

        CombinedFilter = type(
            "CombinedFilter",
            (FilterBase,),
            dict(apply=composition),
        )
        return cast(FilterBase, CombinedFilter())


V = TypeVar("V")


class TrackVariable(Generic[V]):
    def __set_name__(self, owner: FilterBase, name: str) -> None:
        self.public_name = name

        if not hasattr(owner, "tracked_variables"):
            owner.tracked_variables = {}

        owner.tracked_variables[self.public_name] = []

    def __get__(
        self, obj: FilterBase, objtype: None | Type[FilterBase] = None
    ) -> V:
        history = cast(list[V], obj.tracked_variables[self.public_name])
        return history[-1]

    def __set__(self, obj: FilterBase, value: V) -> None:
        obj.tracked_variables[self.public_name].append(value)


class FilterMedian(FilterBase):
    """Rolling median filter"""

    # default kernel values
    default: ClassVar[dict[str, int]] = dict(
        altitude=11,
        geoaltitude=9,
        vertical_rate=5,
        groundspeed=9,
        track=5,
    )

    def __init__(self, **kwargs: int) -> None:
        self.columns = {**self.default, **kwargs}

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        for column, kernel in self.columns.items():
            if column not in data.columns:
                continue
            column_copy = data[column]
            data[column] = data[column].rolling(kernel, center=True).median()
            data.loc[data[column].isnull(), column] = column_copy
        return data


class FilterMean(FilterBase):
    """Rolling mean filter."""

    # default kernel values
    default: ClassVar[dict[str, int]] = dict(
        altitude=10,
        geoaltitude=10,
        vertical_rate=3,
        groundspeed=7,
        track=3,
    )

    def __init__(self, **kwargs: int) -> None:
        self.columns = {**self.default, **kwargs}

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        for column, kernel in self.columns.items():
            if column not in data.columns:
                continue
            column_copy = data[column]
            data[column] = data[column].rolling(kernel, center=True).mean()
            data.loc[data[column].isnull(), column] = column_copy
        return data


class FilterAboveSigmaMedian(FilterBase):
    """Filters noisy values above one sigma wrt median filter.

    The method first applies a median filter on each feature of the
    DataFrame. A default kernel size is applied for a number of features
    (resp. latitude, longitude, altitude, track, groundspeed, IAS, TAS) but
    other kernel values may be passed as kwargs parameters.

    Rather than returning averaged values, the method computes thresholds
    on sliding windows (as an average of squared differences) and replace
    unacceptable values with NaNs.

    Then, a strategy may be applied to fill the NaN values, by default a
    forward/backward fill. Other strategies may be passed, for instance *do
    nothing*: ``None``; or *interpolate*: ``lambda x: x.interpolate()``.

    .. note::

        This method if often more efficient when applied several times with
        different kernel values.Kernel values may be passed as integers, or
        list/tuples of integers for cascade of filters:

        .. code:: python

            # this cascade of filters appears to work well on altitude
            flight.filter(altitude=17).filter(altitude=53)

            # this is equivalent to the default value
            flight.filter(altitude=(17, 53))

    """

    # default kernel values
    default: ClassVar[dict[str, int | tuple[int, ...]]] = dict(
        altitude=(17, 53),
        geoaltitude=(17, 53),
        selected_mcp=(17, 53),
        selected_fms=(17, 53),
        IAS=23,
        TAS=23,
        Mach=23,
        vertical_rate=3,
        groundspeed=5,
        compute_gs=(17, 53),
        compute_track=17,
        track=17,
        onground=3,
    )

    def __init__(self, **kwargs: int | tuple[int, ...]) -> None:
        self.columns = {**self.default, **kwargs}
        self.empty_kwargs = len(kwargs) == 0

    def cascaded_filters(
        self,
        df: pd.DataFrame,
        feature: str,
        kernel_size: int,
        filt: Callable[[pd.Series, int], Any] = signal.medfilt,
    ) -> pd.DataFrame:
        """Produces a mask for data to be discarded.

        The filtering applies a low pass filter (e.g medfilt) to a signal
        and measures the difference between the raw and the filtered signal.

        The average of the squared differences is then produced (sq_eps) and
        used as a threshold for filtering.

        Errors may raised if the kernel_size is too large
        """
        y = df[feature].astype(float)
        y_m = filt(y, kernel_size)
        sq_eps = (y - y_m) ** 2
        return pd.DataFrame(
            {
                "timestamp": df["timestamp"],
                "y": y,
                "y_m": y_m,
                "sq_eps": sq_eps,
                "sigma": np.sqrt(filt(sq_eps, kernel_size)),
            },
            index=df.index,
        )

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = self.columns
        if self.empty_kwargs:
            features = [
                cast(str, feature)
                for feature in data.columns
                if data[feature].dtype
                in [np.float32, np.float64, np.int32, np.int64]
            ]
            columns = {
                **dict((feature, 17) for feature in features),
                **columns,
            }

        for column, kernels in columns.items():
            if column not in data.columns:
                continue

            if isinstance(kernels, int):
                kernels = (kernels,)
            else:
                kernels = tuple(kernels)

            for size in kernels:
                # Prepare each feature for the filtering
                df = self.cascaded_filters(
                    data[["timestamp", column]], column, size
                )

                # Decision to accept/reject data points in the time series
                data.loc[df.sq_eps > df.sigma, column] = None

        return data


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
            ).fillna(False, inplace=False)

            spike_time = pd.Series(np.nan, index=data.index)
            spike_time.loc[spike] = data[self.time_column].loc[spike]

            if not spike_time.isnull().all():
                spike_time_prev = spike_time.ffill()
                spike_delta_prev = data["timestamp"] - spike_time_prev
                spike_time_next = spike_time.bfill()
                spike_delta_next = spike_time_next - data["timestamp"]
                in_window = np.bitwise_and(
                    spike_delta_prev.dt.total_seconds() <= window,
                    spike_delta_next.dt.total_seconds() <= window,
                )
                data.loc[(in_window), column] = np.NaN

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
                data["group"] = bigdiff.eq(True).cumsum()
                groups = data[data[column].notna()]["group"].value_counts()
                keepers = groups[groups > param["group_size"]].index.tolist()
                data[column] = data[column].where(
                    data["group"].isin(keepers), float("NaN")
                )

        return data.drop(columns=["group"])


class ProcessXYZFilterBase(FilterBase):
    def __init__(self, reject_sigma: int = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        alt: Annotated[Any, "ft"] = df.altitude
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        track: Annotated[Any, "radians"] = np.radians(90 - df.track)
        vertical_rate: Annotated[Any, "ft/min"] = df.vertical_rate

        velocity: Annotated[Any, "m/s"] = groundspeed

        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y
        z: Annotated[Any, "m"] = alt

        dx: Annotated[Any, "m/s"] = velocity * np.cos(track)
        dy: Annotated[Any, "m/s"] = velocity * np.sin(track)
        dz: Annotated[Any, "m/s"] = vertical_rate

        return pd.DataFrame(
            {"x": x, "y": y, "z": z, "dx": dx, "dy": dy, "dz": dz}
        )

    @impunity
    def postprocess(
        self, df: pd.DataFrame
    ) -> Dict[str, npt.NDArray[np.float64]]:
        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y
        z: Annotated[Any, "m"] = df.z

        dx: Annotated[Any, "m/s"] = df.dx
        dy: Annotated[Any, "m/s"] = df.dy
        dz: Annotated[Any, "m/s"] = df.dz

        velocity: Annotated[Any, "m/s"] = np.sqrt(dx**2 + dy**2)
        track = 90 - np.degrees(np.arctan2(dy, dx))
        track = track % 360

        altitude: Annotated[Any, "ft"] = z
        groundspeed: Annotated[Any, "kts"] = velocity
        vertical_rate: Annotated[Any, "ft/min"] = dz

        return dict(
            x=x,
            y=y,
            altitude=altitude,
            groundspeed=groundspeed,
            track=track,
            vertical_rate=vertical_rate,
        )


class KalmanFilter6D(ProcessXYZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_pre: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id6 = np.eye(df.shape[1])
        dt = 1

        self.x_mes = df.iloc[0].values
        self.x_cor = df.iloc[0].values
        self.p_pre = _id6 * 1e5
        self.p_cor = _id6 * 1e5

        R = (
            np.diag(
                [
                    (df.x - df.x.rolling(17).mean()).std(),
                    (df.y - df.y.rolling(17).mean()).std(),
                    (df.z - df.z.rolling(17).mean()).std(),
                    (df.dx - df.dx.rolling(17).mean()).std(),
                    (df.dy - df.dy.rolling(17).mean()).std(),
                    (df.dz - df.dz.rolling(17).mean()).std(),
                ]
            )
            ** 2
        )

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        Q = 1e-1 * np.diag([0.25, 0.25, 0.25, 1, 1, 1]) * R
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=3)
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=-3)

        for i in range(1, df.shape[0]):
            # measurement
            self.x_mes = df.iloc[i].values
            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            A = np.eye(6) + dt * np.eye(6, k=3)
            x_pre = A @ self.x_cor
            p_pre = A @ self.p_cor @ A.T + Q
            H = _id6.copy()

            # DEBUG: p_pre should be symmetric
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Logging the final value...
            self.p_pre = p_pre

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x_cor = x_pre + K @ nu

            # covariance correction
            self.p_cor = (_id6 - K @ H) @ p_pre @ (_id6 - K @ H).T + K @ R @ K.T

            # DEBUG: p_cor should be symmetric
            # assert np.abs(self.p_cor - self.p_cor.T).sum() < 1e-6

        filtered = pd.DataFrame(
            self.tracked_variables["x_cor"],
            columns=["x", "y", "z", "dx", "dy", "dz"],
        )

        return data.assign(**self.postprocess(filtered))


class KalmanSmoother6D(ProcessXYZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    xs: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id6 = np.eye(df.shape[1])
        dt = 1

        self.x_mes = df.iloc[0].values
        self.x1_cor = df.iloc[0].values
        self.p1_cor = _id6 * 1e5

        R = (
            np.diag(
                [
                    (df.x - df.x.rolling(17).mean()).std(),
                    (df.y - df.y.rolling(17).mean()).std(),
                    (df.z - df.z.rolling(17).mean()).std(),
                    (df.dx - df.dx.rolling(17).mean()).std(),
                    (df.dy - df.dy.rolling(17).mean()).std(),
                    (df.dz - df.dz.rolling(17).mean()).std(),
                ]
            )
            ** 2
        )

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        Q = 1e-1 * np.diag([0.25, 0.25, 0.25, 1, 1, 1]) * R
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=3)
        Q += 0.5 * np.diag(Q.diagonal()[3:], k=-3)

        # >>> First FORWARD  <<<

        for i in range(1, df.shape[0]):
            # measurement
            self.x_mes = df.iloc[i].values

            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            A = np.eye(6) + dt * np.eye(6, k=3)
            x_pre = A @ self.x1_cor
            p_pre = A @ self.p1_cor @ A.T + Q
            H = _id6.copy()

            # DEBUG symmetric matrices
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                # 3 sigma ^2 * nb_dim (6)

                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p1_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x1_cor = x_pre + K @ nu

            # covariance correction
            self.p1_cor = (_id6 - K @ H) @ p_pre @ (
                _id6 - K @ H
            ).T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p1_cor - self.p1_cor.T).sum() < 1e-6

        # >>> Now BACKWARD  <<<
        self.x2_cor = self.x1_cor
        self.p2_cor = 100 * self.p1_cor
        dt = -dt

        for i in range(df.shape[0] - 1, 0, -1):
            # measurement
            self.x_mes = df.iloc[i].values
            # replace NaN values with crazy values
            # they will be filtered out because out of the 3 \sigma enveloppe
            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, 1e24)

            # prediction
            A = np.eye(6) + dt * np.eye(6, k=3)
            x_pre = A @ self.x2_cor
            p_pre = A @ self.p2_cor @ A.T + Q
            H = _id6.copy()

            # DEBUG symmetric matrices
            # assert np.abs(p_pre - p_pre.T).sum() < 1e-6

            # innovation
            nu = x_mes - x_pre
            S = H @ p_pre @ H.T + R

            if (nu.T @ np.linalg.inv(S) @ nu) > self.reject_sigma**2 * 6:
                # 3 sigma ^2 * nb_dim (6)

                sm1 = np.linalg.inv(S)

                x = (sm1 @ nu) * nu

                # identify faulty measurements
                idx = np.where(x > self.reject_sigma**2)

                # replace the measure by the prediction for faulty data
                x_mes[idx] = x_pre[idx]
                nu = x_mes - x_pre

                # ignore the fault data for the covariance update
                H[idx, idx] = 0

                p_pre = A @ self.p2_cor @ A.T + Q
                S = H @ p_pre @ H.T + R

            # Logging the final value...
            self.p_pre = p_pre

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x2_cor = x_pre + K @ nu

            # covariance correction
            self.p2_cor = (_id6 - K @ H) @ p_pre @ (
                _id6 - K @ H
            ).T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        for i in range(1, df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

        filtered = pd.DataFrame(
            self.tracked_variables["xs"],
            columns=["x", "y", "z", "dx", "dy", "dz"],
        )

        return data.assign(**self.postprocess(filtered))



def distance(lat1,lon1,lat2,lon2):
    r = 6371000
    phi1 = lat1
    phi2 = lat2
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return res / NM2METER

def lag(horizon, v):
    res = np.empty((horizon,v.shape[0]))
    res.fill(np.nan)
    for i in range(horizon):
        res[i,:v.shape[0]-i] = v[i:]
    return res

def diffangle(a,b):
    d = a - b
    return d + 2 * np.pi * ((d<-np.pi).astype(float)-(d>=np.pi).astype(float))

def dxdy_from_dlat_dlon(lat_rad, lon_rad, dlat, dlon):
    a = 6378137
    b = 6356752.314245
    e2 = 1 - (b/a)**2
    sinmu2 = np.sin(lat_rad) ** 2
    rm = a * (1-e2)/(1-e2*sinmu2)**(3/2)
    bign = a / np.sqrt(1-e2*sinmu2)
    dx = dlon * bign *  np.cos(lat_rad)
    dy = dlat * rm
    return dx,dy


def compute_gtgraph(dd):
    '''compute the graph of points complying with the speed limits: i and j are adjacent if i can be reached by j within the speed limits'''
    import graph_tool as gt
    n = dd.shape[1]
    # horizon = dd.shape[0]
    g=gt.Graph(g=n,directed=True)#,g=dd.shape[0])
    eprop_dist = g.new_edge_property("int")
    d={i:v for i,v in enumerate(g.vertices())}
    edges = [(i,i+h) for h,i in zip(*np.nonzero(dd)) if h>0]
    g.add_edge_list(edges)#,eprops=eprop_dist)
    eprop_dist = g.new_edge_property("int",val=-1)
    return d,g,eprop_dist

def get_gtlongest(dd):
    '''compute the longest path of points complying with the speed limits'''
    # import graph_tool as gt
    import graph_tool as gt
    v,g,prop_dist = compute_gtgraph(dd)
    n = g.num_vertices()
    vend = g.add_vertex()
    vstart = g.add_vertex()
    for v in g.iter_vertices():
        if v!=vend:
            e=g.add_edge(v,vend)
            prop_dist[e]=-1
        if v!=vstart:
            e=g.add_edge(vstart,v)
            prop_dist[e]=-1
    longest_path,_ = gt.topology.shortest_path(g,source=vstart,target=vend,weights=prop_dist,negative_weights=True,dag=True)
    longest_path =list(map(int,longest_path))
    return longest_path


def get_nxlongest(dd):
    import networkx as nx
    n = dd.shape[1]
    g = nx.DiGraph()
    for i in range(n):
        g.add_node(i)
    edges = [(i,i+h,{"weight":-1}) for h,i in zip(*np.nonzero(dd)) if h>0]
    g.add_edges_from(edges)
    for i in range(n+1):
        g.add_edge(-1,i,weight=-1)
    for i in range(-1,n):
        g.add_edge(i,n,weight=-1)
    path = nx.shortest_path(g,source = -1, target=n,weight="weight",method="bellman-ford")
    return path

def exact_solver(dd):
    try:
        longest_path = get_gtlongest(dd)#,weights)
    except ImportError:
        warnings.warn("graph-tool library not installed, switching to slower NetworkX library")
        longest_path = get_nxlongest(dd)
    res = np.full(dd.shape[1],True)
    res[np.array(longest_path)[1:-1]] = False
    return res

def approx_solver(dd):
    ddbwd = np.empty_like(dd)
    ddbwd.fill(False)
    for h in range(dd.shape[0]):
        assert(dd[h,:dd.shape[1]-h].shape==ddbwd[h,h:].shape)
        ddbwd[h,h:] = dd[h,:dd.shape[1]-h]
    res = np.full(dd.shape[1], True)
    out = 10 * res.shape[0]#30000
    def selectpoints(iinit,dd,opadd,argmaxopadd):
        jumpmin = dd[1:].argmax(axis=0)+1
        jumpmin[(dd[1:,]==False).all(axis=0)==True]=out
        ### computes heuristic
        succeed = np.zeros(res.shape,dtype=np.int64)
        nsteps = 10
        posjumpmin = np.arange(res.shape[0])
        def check_in(pos):
            return np.logical_and(0<=posjumpmin,posjumpmin<res.shape[0])
        for k in range(nsteps):
            valid = check_in(posjumpmin)
            posjumpminvalid = posjumpmin[valid]
            posjumpmin[valid] = opadd(posjumpminvalid, jumpmin[posjumpminvalid])
            succeed[check_in(posjumpmin)]+=1
        posjumpmin[succeed!=nsteps]= opadd(0,out-succeed[succeed!=nsteps])
        #### end compute heuristic
        def selectjump(i, cjump, prediction):
            return cjump[argmaxopadd(prediction[opadd(i,cjump)])]
        i=iinit
        while 0 <= i < res.shape[0]:
            res[i]=False
            candidatejump = np.arange(1,dd.shape[0])[dd[1:,i]]
            if len(candidatejump)==0:
                break
            else:
                i = opadd(i,selectjump(i,candidatejump,posjumpmin))
    ss = np.sum(dd[1:],axis=0)+np.sum(ddbwd[1:],axis=0)
    iinit = ss.argmax()
    selectpoints(iinit,dd,operator.add,np.argmin)
    selectpoints(iinit,ddbwd,operator.sub,np.argmax)
    return res

def consistency_solver(dd, exact_when_kept_below):
    if exact_when_kept_below == 1:
        return exact_solver(dd)
    else:
        mask = approx_solver(dd)
        if 1-np.mean(mask) < exact_when_kept_below:
            return exact_solver(dd)
        else:
            return mask

class FilterConsistency(FilterBase):
    """Filters noisy values, keeping only values consistent with each other. Consistencies are checked between points :math:`i` and points :math:`j\in [|i+1;i+horizon|]`. Using these consistencies, a graph is built: if :math:`i` and :math:`j` are consistent, an edge :math:`(i,j)` is added to the graph. The kept values is the longest path in this graph, resulting in a sequence of consistent values.

    The consistencies checked vertically between :math:`t_i<t_j` are::
    :math:`|(alt_j-alt_i)-(t_j-t_i)* ROCD_i| < clip((t_j-t_i)*dalt\_dt,dalt\_min,dalt\_max)` and
    :math:`|(alt_j-alt_i)-(t_j-t_i)* ROCD_j| < clip((t_j-t_i)*dalt\_dt,dalt\_min,dalt\_max)`
    where :math:`dalt\_dt`, :math:`dalt\_min` and :math:`dalt\_max` are thresholds that can be specified by the user.

    The consistencies checked horizontally between :math:`t_i<t_j` are:
    :math:`|track_i-atan2(lat_j-lat_i,lon_j-lon_i)| < (t_j-t_i)*dtrack\_dt` and
    :math:`|track_j-atan2(lat_j-lat_i,lon_j-lon_i)| < (t_j-t_i)*dtrack\_dt` and
    :math:`|dist(lat_j,lat_i,lon_j,lon_i)-groundspeed_i*(t_j-t_i)| < clip((t_j-t_i)*groundspeed_i*ddist\_dt\_over\_gspeed,0,ddist\_max)`
    where :math:`dtrack\_dt`, :math:`ddist\_dt\_over\_gspeed` and :math:`ddist\_max` are thresholds that can be specified by the user.

    If :math:`horiz\_and\_verti\_together=False` then two graphs are built, one for the vertical variables and one for horizontal variables. Otherwise, only one graph is built. Using two graphs is more precise but slower.

    In order to compute the longest path faster, a greedy algorithm is used. However, if the ratio of kept points is inferior to :math:`exact\_when\_kept\_below` then an exact and slower computation is triggered. This computation uses the Network library or the faster graph-tool library if available.

    This filter replaces unacceptable values with NaNs. Then, a strategy may be applied to fill the NaN values, by default a forward/backward fill. Other strategies may be passed, for instance do nothing: None; or interpolate: lambda x: x.interpolate(limit_area='inside')

    """
    default: ClassVar[dict[str, float | tuple[float, ...]]] = dict(
        dtrack_dt=2, # [degree/s]
        dalt_dt=200,# [feet/min]
        dalt_min=50,# [feet]
        dalt_max=4000,# [feet]
        ddist_dt_over_gspeed=0.2,# [-]
        ddist_max=3,# [NM]
    )
    def __init__(self, horizon: int | None = 200 , horiz_and_verti_together: bool = False, exact_when_kept_below: float = 0.5, **kwargs: float | tuple[float, ...]) -> None:
        self.horiz_and_verti_together = horiz_and_verti_together
        self.horizon = horizon
        self.thresholds = {**self.default, **kwargs}
        self.exact_when_kept_below = exact_when_kept_below
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        lat = data.latitude.values
        lon = data.longitude.values
        alt = data.altitude.values
        rocd = data.vertical_rate.values
        gspeed = data.groundspeed.values
        t = ((data.timestamp-data.timestamp.iloc[0])/pd.to_timedelta(1,unit='s')).values
        assert(t.min()>=0)
        n = lat.shape[0]
        horizon = n if self.horizon is None else min(self.horizon, n)
        def lagh(v):
            return lag(horizon,v)
        dt = abs(lagh(t)-t)/3600
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lag_lat_rad = lagh(lat_rad)
        # mask_nan = np.isnan(lag_lat_rad)
        dlat =  lag_lat_rad - lat_rad
        dlon = lagh(lon_rad) - lon_rad
        dx, dy = dxdy_from_dlat_dlon(lat_rad, lon_rad, dlat, dlon)
        track_rad = np.radians(data.track.values)
        lag_track_rad = lagh(track_rad)
        mytrack_rad = np.arctan2(-dx, -dy) + np.pi
        thresh_track = np.radians(self.thresholds["dtrack_dt"]) * 3600 * dt
        ddtrack_fwd = np.abs(diffangle(mytrack_rad,track_rad)) < thresh_track
        ddtrack_bwd = np.abs(diffangle(mytrack_rad,lag_track_rad)) < thresh_track
        ddtrack = np.logical_and(ddtrack_fwd, ddtrack_bwd)
        dist = distance(lag_lat_rad, lagh(lon_rad), lat_rad, lon_rad)
        absdist = abs(dist)
        gdist_matrix = gspeed * dt
        ddspeed = np.abs(gdist_matrix - absdist) < np.clip(self.thresholds["ddist_dt_over_gspeed"] * gspeed * dt,0,self.thresholds["ddist_max"])
        ddhorizontal = np.logical_and(ddtrack,ddspeed)

        dalt = lagh(alt)-alt
        thresh_alt = np.clip(self.thresholds["dalt_dt"]*60*dt,
                             self.thresholds["dalt_min"],
                             self.thresholds["dalt_max"])
        ddvertical = np.logical_and(np.abs(dalt-rocd*dt*60)<thresh_alt,
                                    np.abs(dalt-lagh(rocd)*dt*60)<thresh_alt)
        if self.horiz_and_verti_together:
            mask_horiz = consistency_solver(np.logical_and(ddhorizontal,ddvertical),
                                            self.exact_when_kept_below)
            mask_verti = mask_horiz
        else:
            mask_horiz = consistency_solver(ddhorizontal, self.exact_when_kept_below)
            mask_verti = consistency_solver(ddvertical, self.exact_when_kept_below)
        print(1-np.mean(mask_horiz))
        print(1-np.mean(mask_verti))
        data.loc[mask_horiz,["track","longitude","latitude","groundspeed"]]=np.NaN
        data.loc[mask_verti,["altitude","vertical_rate"]]=np.NaN
        return data
