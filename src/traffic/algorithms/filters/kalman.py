from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from .preprocessing import (
    ProcessXYZFilterBase,
    TrackVariable,
)


class KalmanFilter6D(ProcessXYZFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p_pre: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

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
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    x2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
    p2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

    xs: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()

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
            imkh = _id6 - K @ H
            self.p2_cor = imkh @ p_pre @ imkh.T + K @ R @ K.T

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
