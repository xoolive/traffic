from __future__ import annotations

from typing import Annotated, Any, Dict, Generic, Type, TypeVar, cast

from impunity import impunity

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.extensions import ExtensionArray

from . import FilterBase

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


class ProcessXYFilterBase(FilterBase):
    """Assistant class to preprocess the dataframe and build features.

    - Expects x and y features.
    - Provides x, y, dx and dy features.
    - Reconstruct groundspeed (in kts) and track angle.
    """

    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        track: Annotated[Any, "radians"] = np.radians(90.0 - df.track)

        velocity: Annotated[Any, "m/s"] = groundspeed

        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y

        dx: Annotated[Any, "m/s"] = velocity * np.cos(track)
        dy: Annotated[Any, "m/s"] = velocity * np.sin(track)

        return pd.DataFrame(
            {
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                "v": velocity,
                "theta": track,
            }
        )

    @impunity
    def postprocess(self, df: pd.DataFrame) -> Dict[str, ExtensionArray]:
        x: Annotated[pd.Series[float], "m"] = df.x
        y: Annotated[pd.Series[float], "m"] = df.y
        velocity: Annotated[pd.Series[float] | npt.NDArray[np.float64], "m/s"]

        if "dx" in df.columns:
            dx: Annotated[pd.Series[float], "m/s"] = df.dx
            dy: Annotated[pd.Series[float], "m/s"] = df.dy

            velocity = np.sqrt(dx**2 + dy**2)
            track = 90.0 - np.degrees(np.arctan2(dy, dx))
            track = track % 360
        else:
            velocity = df.v
            track = 90 - np.degrees(df.theta.values)
            track = track % 360

        groundspeed: Annotated[Any, "kts"] = velocity

        return dict(
            x=x.values,
            y=y.values,
            groundspeed=groundspeed.values,
            track=track,
        )


class ProcessXYZFilterBase(FilterBase):
    """Assistant class to preprocess the dataframe and build features.

    - Expects x and y features.
    - Provides x, y, z, dx, dy and dz features.
    - Reconstruct vertical rate (in ft/min), groundspeed (in kts) and track.
    """

    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        alt: Annotated[Any, "ft"] = df.altitude
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        track: Annotated[Any, "radians"] = np.radians(90.0 - df.track)
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
        x: Annotated[pd.Series[float], "m"] = df.x
        y: Annotated[pd.Series[float], "m"] = df.y
        z: Annotated[pd.Series[float], "m"] = df.z

        dx: Annotated[pd.Series[float], "m/s"] = df.dx
        dy: Annotated[pd.Series[float], "m/s"] = df.dy
        dz: Annotated[pd.Series[float], "m/s"] = df.dz

        velocity: Annotated[pd.Series[float], "m/s"]
        velocity = np.sqrt(dx**2 + dy**2)
        track = 90.0 - np.degrees(np.arctan2(dy, dx))
        track = track % 360

        altitude: Annotated[pd.Series[float], "ft"] = z
        groundspeed: Annotated[pd.Series[float], "kts"] = velocity
        vertical_rate: Annotated[pd.Series[float], "ft/min"] = dz

        return dict(
            x=x.values,
            y=y.values,
            altitude=altitude.values,
            groundspeed=groundspeed.values,
            track=track.values,
            vertical_rate=vertical_rate.values,
        )


class KalmanFilter6D(ProcessXYZFilterBase):
    """A basic Kalman Filter with 6 components.

    The filter requires x, y, z, dx, dy and dz components.
    """

    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    x_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    p_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    p_pre: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()

    def __init__(self, reject_sigma: float = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

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
            x_mes = np.where(~np.isnan(self.x_mes), self.x_mes, 1e24)

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
            imkh = _id6 - K @ H
            self.p_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG: p_cor should be symmetric
            # assert np.abs(self.p_cor - self.p_cor.T).sum() < 1e-6

        filtered = pd.DataFrame(
            self.tracked_variables["x_cor"],
            columns=["x", "y", "z", "dx", "dy", "dz"],
        )

        return data.assign(**self.postprocess(filtered))


class KalmanSmoother6D(ProcessXYZFilterBase):
    """A basic two-pass Kalman smoother with 6 components.

    The filter requires x, y, z, dx, dy and dz components.
    """

    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    x1_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    p1_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    x2_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    p2_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()

    xs: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()

    def __init__(self, reject_sigma: float = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma

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
            x_mes = np.where(~np.isnan(self.x_mes), self.x_mes, 1e24)

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
            imkh = _id6 - K @ H
            self.p1_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

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
            x_mes = np.where(~np.isnan(self.x_mes), self.x_mes, 1e24)

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
            imkh = _id6 - K @ H
            self.p2_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

            # DEBUG symmetric matrices
            # assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        for i in range(df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

        filtered = pd.DataFrame(
            self.tracked_variables["xs"],
            columns=["x", "y", "z", "dx", "dy", "dz"],
        )

        return data.assign(**self.postprocess(filtered))
