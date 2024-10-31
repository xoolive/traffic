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
