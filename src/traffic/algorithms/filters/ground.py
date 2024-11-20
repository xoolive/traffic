from __future__ import annotations

from typing import Any

from cartopy.crs import Projection

import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely import shortest_line
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ...data import airports
from .preprocessing import (
    ProcessXYFilterBase,
    TrackVariable,
)


class KalmanTaxiway(ProcessXYFilterBase):
    # Descriptors are convenient to store the evolution of the process
    x_mes: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    x_pre: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    x1_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    p1_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    x2_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    p2_cor: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()

    xs: TrackVariable[pd.core.arrays.ExtensionArray] = TrackVariable()
    shl: TrackVariable[Any] = TrackVariable()
    closest_line: TrackVariable[Any] = TrackVariable()

    # todo clean that
    columns: Any
    postprocess: Any

    def __init__(
        self,
        airport: str,
        projection: Projection,
        closest: None | list[BaseGeometry] = None,
        option: str = "xs",
    ) -> None:
        super().__init__()

        # # bruit de mesure
        # self.R = np.diag([9, 9, 2, 2, 1]) ** 2

        # # plus de bruit de modèle sur les dérivées qui sont pas recalées
        # self.Q = 0.5 * np.diag([0.25, 0.25, 2, 2])
        # self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=2)
        # self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=-2)

        self.columns = ["x", "y", "dx", "dy"]

        # bruit de mesure
        self.R = np.diag([4, 4, 2, 2, 1]) ** 2

        # plus de bruit de modèle sur les dérivées qui sont pas recalées
        self.Q = np.diag([0.25, 0.25, 2, 2])
        self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=2)
        self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=-2)

        self.projection = projection
        self.taxiways = (
            airports[airport]
            ._openstreetmap()
            .query(
                'aeroway=="taxiway" or aeroway == "runway" or '
                'aeroway == "parking_position" or aeroway == "apron" '
            )
            .assign(geom_type=lambda df: df.geom_type)
            .query('geom_type == "LineString"')
            .data.set_crs(epsg=4326)
            .to_crs(self.projection.proj4_init)
        )
        self.closest = closest
        if closest is not None:
            self.taxiways = self.taxiways.iloc[sorted(set(closest))]

        self.option = option

    def distance(
        self, prediction: npt.NDArray[np.float64], idx: int
    ) -> tuple[float, float, float]:
        point_mes = Point(*self.x_mes[:2])
        point_pre = Point(*prediction[:2])

        if False:  # self.closest is not None:
            idx = self.closest_line = self.closest[idx]
        else:
            if (self.x_mes[:2] != self.x_mes[:2]).any():
                distance_to_taxiway = self.taxiways.distance(point_pre)
            else:
                distance_to_taxiway = self.taxiways.distance(point_mes)

            idx = self.closest_line = distance_to_taxiway.argmin()

        closest_line = self.taxiways.iloc[idx].geometry
        self.shl = shortest_line(closest_line, point_pre)

        if self.shl is None:
            return (0, 0, 0)

        distance = self.shl.length
        (x1, y1), (x2, y2) = self.shl.coords
        dx = (x2 - x1) / distance if distance else 0
        dy = (y2 - y1) / distance if distance else 0

        # in order, distance, dx, dy
        return distance, dx, dy

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocess(data)[["x", "y", "dx", "dy"]]

        for cumul in self.tracked_variables.values():
            cumul.clear()

        # initial state
        _id4 = np.eye(df.shape[1])
        dt = 1

        self.x_mes = df.iloc[0].values
        self.x_pre = self.x_mes
        self.x1_cor = df.iloc[0].values
        self.p1_cor = _id4 * 1e5

        # >>> First FORWARD  <<<

        for i in range(1, df.shape[0]):
            # measurement
            self.x_mes = df.iloc[i].values

            # prediction
            A = np.eye(4) + dt * np.eye(4, k=2)
            x1_cor = self.x1_cor
            x1_cor[2:] = np.where(x1_cor[2:] == x1_cor[2:], x1_cor[2:], 0)

            x_pre = A @ x1_cor
            self.x_pre = x_pre
            p_pre = A @ self.p1_cor @ A.T + self.Q

            distance, dx, dy = self.distance(x_pre, i)

            H = np.r_[_id4.copy(), np.array([[dx, dy, 0, 0]])]

            # DEBUG symmetric matrices
            assert np.abs(p_pre - p_pre.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(p_pre) > 0)

            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, x_pre)
            idx = np.where(self.x_mes != self.x_mes)
            H[idx, idx] = 0

            # innovation
            nu = np.r_[x_mes - x_pre, [-distance]]
            S = H @ p_pre @ H.T + self.R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)
            # state correction
            self.x1_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id4 - K @ H
            self.p1_cor = imkh @ p_pre @ imkh.T + K @ self.R @ K.T

            # DEBUG symmetric matrices
            assert np.abs(self.p1_cor - self.p1_cor.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(self.p1_cor) > 0)

        # >>> Now BACKWARD  <<<
        self.x2_cor = self.x1_cor
        self.p2_cor = 100 * self.p1_cor
        dt = -dt

        for i in range(df.shape[0] - 1, 0, -1):
            # measurement
            self.x_mes = df.iloc[i].values

            # prediction
            A = np.eye(4) + dt * np.eye(4, k=2)
            x_pre = A @ self.x2_cor
            p_pre = A @ self.p2_cor @ A.T + self.Q

            distance, dx, dy = self.distance(x_pre, i)

            H = np.r_[_id4.copy(), np.array([[dx, dy, 0, 0]])]

            # DEBUG symmetric matrices
            assert np.abs(p_pre - p_pre.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(p_pre) > 0)

            x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, x_pre)
            idx = np.where(self.x_mes != self.x_mes)
            H[idx, idx] = 0

            # innovation
            nu = np.r_[x_mes - x_pre, [-distance]]
            S = H @ p_pre @ H.T + self.R

            # Kalman gain
            K = p_pre @ H.T @ np.linalg.inv(S)

            # state correction
            self.x2_cor = x_pre + K @ nu

            # covariance correction
            imkh = _id4 - K @ H
            self.p2_cor = imkh @ p_pre @ imkh.T + K @ self.R @ K.T

            # DEBUG symmetric matrices
            assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6
            assert np.all(np.linalg.eigvals(self.p2_cor) > 0)

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        for i in range(0, df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

        filtered = pd.DataFrame(
            self.tracked_variables["xs"],
            columns=["x", "y", "dx", "dy"],
        )

        df = data  # just because it's missing
        self.tracked_variables["xs"].clear()

        # and the smoothing
        x1_cor = np.array(self.tracked_variables["x1_cor"])
        p1_cor = np.array(self.tracked_variables["p1_cor"])
        x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
        p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])

        more_probable = []
        for i in range(0, df.shape[0]):
            s1 = np.linalg.inv(p1_cor[i])
            s2 = np.linalg.inv(p2_cor[i])
            self.ps = np.linalg.inv(s1 + s2)
            self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T

            score_1 = np.linalg.eigvals(p1_cor[i]).mean()
            score_2 = np.linalg.eigvals(p2_cor[i]).mean()

            if score_1 < score_2:
                more_probable.append(x1_cor[i])
            else:
                more_probable.append(x2_cor[i])

        filtered = pd.DataFrame(
            more_probable
            if self.option == "max"
            else self.tracked_variables[self.option],
            columns=self.columns,
        )

        return data.assign(**self.postprocess(filtered))


class BaseAirportFilter: ...


# class EKFTaxiway(ProcessXYFilterBase):
#     # Descriptors are convenient to store the evolution of the process
#     x_mes: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
#     x1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
#     p1_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
#     x2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
#     p2_cor: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
#
#     xs: TrackVariable[npt.NDArray[np.float64]] = TrackVariable()
#     shl: TrackVariable[Any] = TrackVariable()
#     closest_line: TrackVariable[Any] = TrackVariable()
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         # bruit de mesure
#         self.R = np.diag([9, 9, 25, 2, 0.25]) ** 2
#
#         # plus de bruit de modèle sur les dérivées qui sont pas recalées
#         self.Q = 0.5 * np.diag([0.25, 0.25, 2, 2])
#         self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=2)
#         self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=-2)
#
#     def distance(
#         self, prediction: npt.NDArray[np.float64], idx: int
#     ) -> tuple[float, float, float]:
#         raise NotImplementedError
#
#     def apply(self, data: pd.DataFrame) -> pd.DataFrame:
#         df = self.preprocess(data)[["x", "y", "v", "theta"]]
#
#         for cumul in self.tracked_variables.values():
#             cumul.clear()
#
#         # initial state
#         _id4 = np.eye(df.shape[1])
#         dt = 1
#
#         self.x_mes = df.iloc[0].values
#         self.x1_cor = df.iloc[0].values
#         self.p1_cor = _id4 * 1e5
#
#         # >>> First FORWARD  <<<
#
#         for i in range(1, df.shape[0]):
#             # measurement
#             self.x_mes = df.iloc[i].values
#
#             x1_cor = self.x1_cor
#             v, theta = x1_cor[2:] = np.where(
#                 x1_cor[2:] == x1_cor[2:], x1_cor[2:], 0
#             )
#
#             # prediction
#             F = np.eye(4)
#             F[0, 2] = dt * np.cos(theta)
#             F[1, 2] = dt * np.sin(theta)
#
#             x_pre = F @ x1_cor
#
#             jF = np.eye(4)
#             jF[:2, 2:] = [
#                 [dt * np.cos(theta), -v * dt * np.sin(theta)],
#                 [dt * np.sin(theta), v * dt * np.cos(theta)],
#             ]
#
#             p_pre = jF @ self.p1_cor @ jF.T + self.Q
#
#             distance, dx, dy = self.distance(x_pre, i)
#
#             H = np.r_[_id4.copy(), np.array([[dx, dy, 0, 0]])]
#
#             # DEBUG symmetric matrices
#             assert np.abs(p_pre - p_pre.T).sum() < 1e-6
#             assert np.all(np.linalg.eigvals(p_pre) > 0)
#
#             x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, x_pre)
#             idx = np.where(self.x_mes != self.x_mes)
#             H[idx, idx] = 0
#
#             # innovation
#             nu = np.r_[x_mes - x_pre, [-distance]]
#             S = H @ p_pre @ H.T + self.R
#
#             # Kalman gain
#             K = p_pre @ H.T @ np.linalg.inv(S)
#             # state correction
#             self.x1_cor = x_pre + K @ nu
#
#             # covariance correction
#             imkh = _id4 - K @ H
#             self.p1_cor = imkh @ p_pre @ imkh.T + K @ self.R @ K.T
#
#             # DEBUG symmetric matrices
#             assert np.abs(self.p1_cor - self.p1_cor.T).sum() < 1e-6
#             assert np.all(np.linalg.eigvals(self.p1_cor) > 0)
#
#         # >>> Now BACKWARD  <<<
#         self.x2_cor = self.x1_cor
#         self.p2_cor = 100 * self.p1_cor
#         dt = -dt
#
#         for i in range(df.shape[0] - 1, 0, -1):
#             # measurement
#             self.x_mes = df.iloc[i].values
#
#             x2_cor = self.x2_cor
#             v, theta = x2_cor[2:] = np.where(
#                 x2_cor[2:] == x2_cor[2:], x2_cor[2:], 0
#             )
#
#             # prediction
#             F = np.eye(4)
#             F[0, 2] = dt * np.cos(theta)
#             F[1, 2] = dt * np.sin(theta)
#
#             x_pre = F @ self.x2_cor
#
#             jF = np.eye(4)
#             jF[:2, 2:] = [
#                 [dt * np.cos(theta), -v * dt * np.sin(theta)],
#                 [dt * np.sin(theta), v * dt * np.cos(theta)],
#             ]
#
#             p_pre = jF @ self.p2_cor @ jF.T + self.Q
#
#             distance, dx, dy = self.distance(x_pre, i)
#
#             H = np.r_[_id4.copy(), np.array([[dx, dy, 0, 0]])]
#
#             # DEBUG symmetric matrices
#             assert np.abs(p_pre - p_pre.T).sum() < 1e-6
#             assert np.all(np.linalg.eigvals(p_pre) > 0)
#
#             x_mes = np.where(self.x_mes == self.x_mes, self.x_mes, x_pre)
#             idx = np.where(self.x_mes != self.x_mes)
#             H[idx, idx] = 0
#
#             # innovation
#             nu = np.r_[x_mes - x_pre, [-distance]]
#             S = H @ p_pre @ H.T + self.R
#
#             # Kalman gain
#             K = p_pre @ H.T @ np.linalg.inv(S)
#
#             # state correction
#             self.x2_cor = x_pre + K @ nu
#
#             # covariance correction
#             imkh = _id4 - K @ H
#             self.p2_cor = imkh @ p_pre @ imkh.T + K @ self.R @ K.T
#
#             # DEBUG symmetric matrices
#             assert np.abs(self.p2_cor - self.p2_cor.T).sum() < 1e-6
#             assert np.all(np.linalg.eigvals(self.p2_cor) > 0)
#
#         # and the smoothing
#         x1_cor = np.array(self.tracked_variables["x1_cor"])
#         p1_cor = np.array(self.tracked_variables["p1_cor"])
#         x2_cor = np.array(self.tracked_variables["x2_cor"][::-1])
#         p2_cor = np.array(self.tracked_variables["p2_cor"][::-1])
#
#         for i in range(0, df.shape[0]):
#             s1 = np.linalg.inv(p1_cor[i])
#             s2 = np.linalg.inv(p2_cor[i])
#             self.ps = np.linalg.inv(s1 + s2)
#             self.xs = (self.ps @ (s1 @ x1_cor[i] + s2 @ x2_cor[i])).T
#
#         filtered = pd.DataFrame(
#             self.tracked_variables["xs"],
#             columns=["x", "y", "v", "theta"],
#         )
#
#         return data.assign(**self.postprocess(filtered))

# class EKFAirport(BaseAirportFilter, EKFTaxiway):
#     def __init__(
#         self,
#         airport: str,
#         projection: Projection,
#         closest: None | list[BaseGeometry] = None,
#         option: str = "xs",
#     ) -> None:
#         super().__init__(airport, projection, closest, option)
#
#         self.columns = ["x", "y", "v", "theta"]
#         # override R and Q if needed
#
#         # bruit de mesure
#         self.R = np.diag([9, 9, 9, 1, 0.25]) ** 2
#
#         # plus de bruit de modèle sur les dérivées qui sont pas recalées
#         self.Q = np.diag([0.25, 0.25, 2, 0.5])
#         self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=2)
#         self.Q += 0.5 * np.diag(self.Q.diagonal()[2:], k=-2)
