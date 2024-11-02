from __future__ import annotations

from typing import Annotated, Any, Callable

from impunity import impunity
from scipy import linalg

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import FilterBase


def extended_kalman_filter(
    measurements: pd.DataFrame,
    initial_state: pd.Series,
    initial_covariance: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    jacobian_state_transition: Callable[
        [pd.Series, float], npt.NDArray[np.float64]
    ],
    state_transition_function: Callable[[pd.Series, float], pd.Series],
    reject_sigma: float = 3,
) -> pd.DataFrame:
    num_states = len(initial_state)
    states = np.repeat(
        initial_state.to_numpy().reshape(1, -1), measurements.shape[0], axis=0
    )
    covariances = np.zeros((measurements.shape[0], num_states, num_states))

    x = initial_state
    P = initial_covariance
    timestamps = measurements.index.to_series()

    for i in range(1, len(timestamps)):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()

        # Prediction Step
        F = jacobian_state_transition(x, dt)
        # Predicted (a priori) state estimate:
        x = state_transition_function(x, dt)
        # Predicted (a priori) estimate covariance:
        P = F @ P @ F.T + Q

        # Measurement update with rejection mechanism
        measurement = measurements.iloc[i]
        H = np.eye(num_states)
        # Innovation or measurement pre-fit residual:
        nu = measurement - x
        # Innovation (or pre-fit residual) covariance:
        S = H @ P @ H.T + R
        std_devs = np.sqrt(np.diag(S))

        # Component-wise standard deviation check (gating)
        for j in range(num_states):
            if abs(nu[j]) > abs(reject_sigma * std_devs[j]):
                measurement.iloc[j] = x.iloc[j]  # Replace faulty measurement
                H[j, j] = 0  # Ignore this component in the update

        # Here we recompute the matrix in case something was rejected:
        # Innovation (or pre-fit residual) covariance:
        S = H @ P @ H.T + R
        # Optimal Kalman gain:
        K = linalg.solve(S, H @ P, assume_a="pos").T
        # Updated (a posteriori) state estimate:
        x = x + K @ nu
        # Updated (a posteriori) estimate covariance:
        P = (np.eye(num_states) - K @ H) @ P

        states[i] = x
        covariances[i] = P

    return pd.DataFrame(states, columns=measurements.columns), covariances


def rts_smoother(
    states: pd.DataFrame,
    covariances: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    timestamps: pd.Series,
    jacobian_state_transition: Callable[
        [pd.Series, float], npt.NDArray[np.float64]
    ],
    state_transition_function: Callable[[pd.Series, float], pd.Series],
) -> pd.DataFrame:
    num_time_steps = states.shape[0]
    smoothed_states = states.copy()
    smoothed_covariances = covariances.copy()

    for i in range(num_time_steps - 2, -1, -1):
        dt = (timestamps[i + 1] - timestamps[i]).total_seconds()

        F = jacobian_state_transition(states.iloc[i], dt)

        # predicted state
        predicted_state = state_transition_function(states.iloc[i], dt)
        # predicted covariance
        Pp = F @ covariances[i] @ F.T + Q

        # G = linalg.solve(Pp, F @ covariances[i], assume_a="pos").T
        G = covariances[i] @ F.T @ np.linalg.inv(Pp)
        smoothed_states.iloc[i] = states.iloc[i] + G @ (
            states.iloc[i + 1] - predicted_state
        )
        smoothed_covariances[i] = (
            covariances[i] + G @ (covariances[i + 1] - Pp) @ G.T
        )

    return smoothed_states


class ProcessXYZZFilterBase(FilterBase):
    @impunity
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        groundspeed: Annotated[Any, "kts"] = df.groundspeed
        # the angle is wrapped between 0 and 2pi, we need to unwrap it
        math_angle: Annotated[Any, "radians"] = np.unwrap(
            np.radians(90 - df.track)
        )
        velocity: Annotated[Any, "m/s"] = groundspeed

        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y

        altitude: Annotated[Any, "ft"] = df.altitude
        # geoaltitude: Annotated[Any, "ft"] = df.geoaltitude
        alt_baro: Annotated[Any, "m"] = altitude
        # alt_geo: Annotated[Any, "m"] = geoaltitude

        vertical_rate: Annotated[Any, "ft/min"] = df.vertical_rate
        vert_rate: Annotated[Any, "m/s"] = vertical_rate

        return pd.DataFrame(
            {
                "x": x,
                "y": y,
                "alt_baro": alt_baro,
                # "alt_geo": alt_geo,
                "math_angle": math_angle,
                "velocity": velocity,
                "vert_rate": vert_rate,
            }
        ).set_index(df["timestamp"])

    @impunity
    def postprocess(
        self, df: pd.DataFrame
    ) -> dict[str, npt.NDArray[np.float64]]:
        x: Annotated[Any, "m"] = df.x
        y: Annotated[Any, "m"] = df.y
        alt_baro: Annotated[Any, "m"] = df.alt_baro
        # alt_geo: Annotated[Any, "m"] = df.alt_geo
        math_angle: Annotated[Any, "radians"] = df.math_angle
        velocity: Annotated[Any, "m/s"] = df.velocity
        vert_rate: Annotated[Any, "m/s"] = df.vert_rate

        altitude: Annotated[Any, "ft"] = alt_baro
        # geoaltitude: Annotated[Any, "ft"] = alt_geo
        track: Annotated[Any, "degree"] = (90 - np.degrees(math_angle)) % 360
        groundspeed: Annotated[Any, "kts"] = velocity
        vertical_rate: Annotated[Any, "ft/min"] = vert_rate

        return dict(
            x=x,
            y=y,
            altitude=altitude,
            # geoaltitude=geoaltitude,
            track=track,
            groundspeed=groundspeed,
            vertical_rate=vertical_rate,
        )


class EKF(ProcessXYZZFilterBase):
    @staticmethod
    def state_transition_function(state: pd.Series, dt: float) -> pd.Series:
        # Unpack the state vector
        # _x, _y, _alt_baro, _alt_geo, math_angle, velocity, vert_rate = state
        _x, _y, _alt_baro, math_angle, velocity, vert_rate = state

        # Compute the derivatives
        x_dot = velocity * np.cos(math_angle)
        y_dot = velocity * np.sin(math_angle)
        altitude_dot = vert_rate
        # geoaltitude_dot = vert_rate

        # Compute the predicted state
        state_pred = state.copy()
        state_pred.loc["x"] += x_dot * dt
        state_pred.loc["y"] += y_dot * dt
        state_pred.loc["alt_baro"] += altitude_dot * dt
        # state_pred.loc["alt_geo"] += geoaltitude_dot * dt
        # Other state variables (math_angle, velocity, vert_rate) are assumed
        # to be constant over the time step
        return state_pred

    @staticmethod
    def jacobian_state_transition(
        x: pd.Series, dt: float
    ) -> npt.NDArray[np.float64]:
        # Unpack the state vector
        # _, _, _, _, math_angle, velocity, _ = x
        _, _, _, math_angle, velocity, _ = x

        # # Compute the Jacobian matrix
        # F_jacobian = np.eye(7)
        # # Partial derivative of x_dot w.r.t math_angle:
        # F_jacobian[0, 4] = -velocity * np.sin(math_angle) * dt
        # # Partial derivative of x_dot w.r.t velocity:
        # F_jacobian[0, 5] = np.cos(math_angle) * dt
        # # Partial derivative of y_dot w.r.t math_angle:
        # F_jacobian[1, 4] = velocity * np.cos(math_angle) * dt
        # # Partial derivative of y_dot w.r.t velocity:
        # F_jacobian[1, 5] = np.sin(math_angle) * dt
        # # Partial derivative of altitude_dot w.r.t vertical_rate
        # F_jacobian[2, 6] = dt
        # # Partial derivative of geoaltitude_dot w.r.t vertical_rate
        # F_jacobian[3, 6] = dt

        # Compute the Jacobian matrix
        F_jacobian = np.eye(6)
        # Partial derivative of x_dot w.r.t math_angle:
        F_jacobian[0, 3] = -velocity * np.sin(math_angle) * dt
        # Partial derivative of x_dot w.r.t velocity:
        F_jacobian[0, 4] = np.cos(math_angle) * dt
        # Partial derivative of y_dot w.r.t math_angle:
        F_jacobian[1, 3] = velocity * np.cos(math_angle) * dt
        # Partial derivative of y_dot w.r.t velocity:
        F_jacobian[1, 4] = np.sin(math_angle) * dt
        # Partial derivative of altitude_dot w.r.t vertical_rate
        F_jacobian[2, 5] = dt
        # Partial derivative of geoaltitude_dot w.r.t vertical_rate
        # F_jacobian[3, 6] = dt

        return F_jacobian

    def __init__(self, smooth: bool = True, reject_sigma: int = 3) -> None:
        super().__init__()
        self.reject_sigma = reject_sigma
        self.smooth = smooth

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        measurements = self.preprocess(data)

        # initial state
        x0 = measurements.iloc[0]  # Initial state
        # P = np.eye(7)  # Initial covariance
        P = np.eye(6)  # Initial covariance

        std_dev_gps = 0
        std_dev_baro = 0
        window_size = 17
        std_dev_track = 0
        std_dev_gps_speed = 0
        std_dev_baro_speed = 0
        R = (
            np.diag(
                [
                    (
                        (
                            measurements.x
                            - measurements.x.rolling(window_size).mean()
                        ).std()
                        ** 2
                        + std_dev_gps**2
                    )
                    ** 0.5,
                    (
                        (
                            measurements.y
                            - measurements.y.rolling(window_size).mean()
                        ).std()
                        ** 2
                        + std_dev_gps**2
                    )
                    ** 0.5,
                    (
                        (
                            measurements.alt_baro
                            - measurements.alt_baro.rolling(window_size).mean()
                        ).std()
                        ** 2
                        + std_dev_baro**2
                    )
                    ** 0.5,
                    # (
                    #     (
                    #         measurements.alt_geo
                    #         - measurements.alt_geo.rolling(window_size).mean()
                    #     ).std()
                    #     ** 2
                    #     + std_dev_gps**2
                    # )
                    # ** 0.5,
                    (
                        (
                            measurements.math_angle
                            - measurements.math_angle.rolling(
                                window_size
                            ).mean()
                        ).std()
                        ** 2
                        + std_dev_track**2
                    )
                    ** 0.5,
                    (
                        (
                            measurements.velocity
                            - measurements.velocity.rolling(window_size).mean()
                        ).std()
                        ** 2
                        + std_dev_gps_speed**2  #
                    )
                    ** 0.5,
                    (
                        (
                            measurements.vert_rate
                            - measurements.vert_rate.rolling(window_size).mean()
                        ).std()
                        ** 2
                        + std_dev_baro_speed**2
                    )
                    ** 0.5,
                ]
            )
            ** 2
        )

        # Q = np.diag([0.1, 0.1, 0.01, 0.01, 0.3, 1, 0.5]) * R
        Q = np.diag([0.1, 0.1, 0.01, 0.3, 1, 0.5]) * R
        filtered_states, filtered_covariances = extended_kalman_filter(
            measurements=measurements,
            initial_state=x0,
            initial_covariance=P,
            Q=Q,
            R=R,  # type: ignore
            jacobian_state_transition=EKF.jacobian_state_transition,
            state_transition_function=EKF.state_transition_function,
            reject_sigma=self.reject_sigma,
        )
        if self.smooth:
            filtered_states = rts_smoother(
                filtered_states,
                filtered_covariances,
                Q,
                measurements.index,
                EKF.jacobian_state_transition,
                EKF.state_transition_function,
            )

        return data.assign(**self.postprocess(filtered_states))
