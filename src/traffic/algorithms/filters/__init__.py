from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Optional,
    Protocol,
    Type,
    cast,
)

import numpy as np
import pandas as pd
import pyproj

if TYPE_CHECKING:
    from cartopy import crs


class Filter(Protocol):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: ...


class FilterBase(Filter):
    tracked_variables: dict[str, list[Any]]
    projection: None | "crs.Projection" | pyproj.Proj = None

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
            column_copy = data[column] = data[column].astype(float)
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
            column_copy = data[column] = data[column].astype(float)
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
        filt: Optional[Callable[[pd.Series[float], int], Any]] = None,
    ) -> pd.DataFrame:
        """Produces a mask for data to be discarded.

        The filtering applies a low pass filter (e.g medfilt) to a signal
        and measures the difference between the raw and the filtered signal.

        The average of the squared differences is then produced (sq_eps) and
        used as a threshold for filtering.

        Errors may raised if the kernel_size is too large
        """
        if filt is None:
            from scipy import signal

            filt = signal.medfilt

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
                feature
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
