import logging
import re
from datetime import datetime, timedelta
from typing import (Callable, Iterable, Iterator, NamedTuple, Optional, Set,
                    Tuple, Union, cast)

import numpy as np

import pandas as pd
import scipy.signal
from cartopy.mpl.geoaxes import GeoAxesSubplot
from shapely.geometry import LineString, base

from ..core.time import time_or_delta, timelike, to_datetime
from .distance import (DistanceAirport, DistancePointTrajectory, closest_point,
                       guess_airport)
from .mixins import DataFrameMixin, GeographyMixin, ShapelyMixin


def _split(data: pd.DataFrame, value, unit) -> Iterator[pd.DataFrame]:
    diff = data.timestamp.diff().values
    if diff.max() > np.timedelta64(value, unit):
        yield from _split(data.iloc[: diff.argmax()], value, unit)
        yield from _split(data.iloc[diff.argmax() :], value, unit)  # noqa
    else:
        yield data


class Flight(DataFrameMixin, ShapelyMixin, GeographyMixin):
    """Flight is the basic class associated to an aircraft itinerary.

    A Flight is supposed to start at takeoff and end after landing, taxiing and
    parking.

    If the current structure seems to contain many flights, warnings may be
    raised.
    """

    def __add__(self, other):
        """Concatenates two Flight objects in the same Traffic structure."""
        if other == 0:
            # useful for compatibility with sum() function
            return self

        # keep import here to avoid recursion
        from .traffic import Traffic

        return Traffic.from_flights([self, other])

    def __radd__(self, other):
        """Concatenates two Flight objects in the same Traffic structure."""
        return self + other

    def _info_html(self) -> str:
        title = f"<b>Flight {self.callsign}</b>"
        if self.number is not None:
            title += f" / {self.number}"
        if self.flight_id is not None:
            title += f" ({self.flight_id})"

        title += "<ul>"
        title += f"<li><b>aircraft:</b> {self.aircraft}</li>"
        if self.origin is not None:
            title += f"<li><b>origin:</b> {self.origin} ({self.start})</li>"
        else:
            title += f"<li><b>origin:</b> {self.start}</li>"
        if self.destination is not None:
            title += f"<li><b>destination:</b> {self.destination} "
            title += f"({self.stop})</li>"
        else:
            title += f"<li><b>destination:</b> {self.stop}</li>"
        title += "</ul>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    @property
    def timestamp(self) -> Iterator[pd.Timestamp]:
        """Iterates the timestamp column of the DataFrame."""
        yield from self.data.timestamp

    @property
    def start(self) -> pd.Timestamp:
        """Returns the minimum timestamp value of the DataFrame."""
        return min(self.timestamp)

    @property
    def stop(self) -> pd.Timestamp:
        """Returns the maximum timestamp value of the DataFrame."""
        return max(self.timestamp)

    @property
    def callsign(self) -> Union[str, Set[str]]:
        """Returns the unique callsign value(s) of the DataFrame."""
        tmp = set(self.data.callsign)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several callsigns for one flight, consider splitting")
        return tmp

    @property
    def number(self) -> Optional[Union[str, Set[str]]]:
        """Returns the unique number value(s) of the DataFrame."""
        if "number" not in self.data.columns:
            return None
        tmp = set(self.data.number)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several numbers for one flight, consider splitting")
        return tmp

    @property
    def icao24(self) -> Union[str, Set[str]]:
        """Returns the unique icao24 value(s) of the DataFrame.

        icao24 is a unique identifier associated to a transponder.
        """
        tmp = set(self.data.icao24)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several icao24 for one flight, consider splitting")
        return tmp

    @property
    def flight_id(self) -> Optional[Union[str, Set[str]]]:
        """Returns the unique flight_id value(s) of the DataFrame.

        If you know how to split flights, you may want to append such a column
        in the DataFrame.
        """
        if "flight_id" not in self.data.columns:
            return None
        tmp = set(self.data.flight_id)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several ids for one flight, consider splitting")
        return tmp

    @property
    def squawk(self) -> Set[int]:
        """Returns all the unique squawk values in the trajectory."""
        return set(self.data.squawk.astype(int))

    @property
    def origin(self) -> Optional[Union[str, Set[str]]]:
        """Returns the unique origin value(s) of the DataFrame.

        The origin airport is mostly represented as a ICAO or a IATA code.
        """
        if "origin" not in self.data.columns:
            return None
        tmp = set(self.data.origin)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several origins for one flight, consider splitting")
        return tmp

    @property
    def destination(self) -> Optional[Union[str, Set[str]]]:
        """Returns the unique destination value(s) of the DataFrame.

        The destination airport is mostly represented as a ICAO or a IATA code.
        """
        if "destination" not in self.data.columns:
            return None
        tmp = set(self.data.destination)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several destinations for one flight, consider splitting")
        return tmp

    def guess_takeoff_airport(self) -> DistanceAirport:
        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[0])

    def guess_landing_airport(self) -> DistanceAirport:
        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[-1])

    def guess_landing_runway(
        # TODO Specify a 'Point' trait in place of NamedTuple
        self,
        airport: Union[None, str, NamedTuple] = None,
    ) -> DistancePointTrajectory:

        from ..data import runways
        from ..data.basic.airport import Airport

        if airport is None:
            airport = self.guess_landing_airport().airport
        if isinstance(airport, Airport):
            airport = airport.icao
        all_runways = runways[airport].values()

        subset = (
            self.airborne()
            .query('vertical_rate < 0')
            .last(minutes=10)
            .resample()
        )
        candidate = subset.closest_point(all_runways)

        avg_track = subset.data.track.tail(10).mean()
        # TODO compute rwy track in the data module
        rwy_track = 10 * int(next(re.finditer('\d+', candidate.name)).group())

        if abs(avg_track - rwy_track) > 20:
            logging.warn(f"({self.flight_id}) Candidate runway "
                         f"{candidate.name} is not consistent "
                         f"with average track {avg_track}.")

        return candidate

    def closest_point(self, points: Union[Iterable[NamedTuple], NamedTuple]):
        # if isinstance(points, NamedTuple):
        if getattr(points, "_asdict", None) is not None:
            points = [points]  # type: ignore
        return min(closest_point(self.data, point) for point in points)

    @property
    def aircraft(self) -> Optional[str]:
        if not isinstance(self.icao24, str):
            return None
        from ..data import aircraft as acdb

        ac = acdb[self.icao24]
        if ac.shape[0] != 1:
            return self.icao24
        else:
            return f"{self.icao24} / {ac.iloc[0].regid} ({ac.iloc[0].mdl})"

    @property
    def registration(self) -> Optional[str]:
        from ..data import aircraft as acdb

        if not isinstance(self.icao24, str):
            return None
        ac = acdb[self.icao24]
        if ac.shape[0] != 1:
            return None
        return ac.iloc[0].regid

    def coords4d(
        self, delta_t: bool = False
    ) -> Iterator[Tuple[float, float, float, float]]:
        data = self.data[self.data.longitude.notnull()]
        altitude = (
            "baro_altitude"
            if "baro_altitude" in self.data.columns
            else "altitude"
        )

        if delta_t:
            time = (data["timestamp"] - data["timestamp"].min()).apply(
                lambda x: x.total_seconds()
            )
        else:
            time = data["timestamp"]

        yield from zip(
            time, data["longitude"], data["latitude"], data[altitude]
        )

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        """Iterates on longitudes, latitudes and altitudes.

        If the baro_altitude field is present, it is preferred over altitude
        """
        data = self.data[self.data.longitude.notnull()]
        altitude = (
            "baro_altitude"
            if "baro_altitude" in self.data.columns
            else "altitude"
        )
        yield from zip(data["longitude"], data["latitude"], data[altitude])

    @property
    def xy_time(self) -> Iterator[Tuple[float, float, float]]:
        """Iterates on longitudes, latitudes and timestamps."""
        iterator = iter(zip(self.coords, self.timestamp))
        while True:
            next_ = next(iterator, None)
            if next_ is None:
                return
            coords, time = next_
            yield (coords[0], coords[1], time.to_pydatetime().timestamp())

    @property
    def linestring(self) -> Optional[LineString]:
        coords = list(self.coords)
        if len(coords) < 2:
            return None
        return LineString(coords)

    @property
    def shape(self) -> Optional[LineString]:
        return self.linestring

    def airborne(self) -> "Flight":
        """Returns the airborne part of the Flight.

        The airborne part is determined by null values on the altitude (or
        baro_altitude if present) column.
        """
        altitude = (
            "baro_altitude"
            if "baro_altitude" in self.data.columns
            else "altitude"
        )
        return self.__class__(self.data[self.data[altitude].notnull()])

    def first(self, **kwargs) -> "Flight":
        return Flight(
            self.data[
                np.stack(self.timestamp) - self.start < timedelta(**kwargs)
            ]
        )

    def last(self, **kwargs) -> "Flight":
        return Flight(
            self.data[
                self.stop - np.stack(self.timestamp) < timedelta(**kwargs)
            ]
        )

    def filter(
        self,
        features: Optional[Iterable[str]] = None,
        kernels_size: Optional[Iterable[int]] = None,
        strategy: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = lambda x: x.bfill().ffill(),
    ) -> "Flight":

        default_kernels_size = {
            "altitude": 17,
            "track": 5,
            "ground_speed": 5,
            "longitude": 15,
            "latitude": 15,
            "cas": 5,
            "tas": 5,
        }

        def cascaded_filters(
            df, feature: str, kernel_size: int, filt=scipy.signal.medfilt
        ) -> pd.DataFrame:
            """Produces a mask for data to be discarded.

            The filtering applies a low pass filter (e.g medfilt) to a signal
            and measures the difference between the raw and the filtered signal.

            The average of the squared differences is then produced (sq_eps) and
            used as a threashold for filtering.

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
                    # useful for now but should be helpful nonetheless
                    "sigma": np.sqrt(filt(sq_eps, kernel_size)),
                },
                index=df.index,
            )

        new_data = self.data.sort_values(by="timestamp").copy()

        if features is None:
            features = [
                cast(str, feature)
                for feature in self.data.columns
                if self.data[feature].dtype
                in [np.float32, np.float64, np.int32, np.int64]
            ]

        if kernels_size is None:
            kernels_size = [0 for _ in features]
            for idx, feature in enumerate(features):
                kernels_size[idx] = default_kernels_size.get(feature, 17)

        for feat, ks in zip(features, kernels_size):

            # Prepare each flight for the filtering
            df = cascaded_filters(new_data[["timestamp", feat]], feat, ks)

            # Decision to accept/reject for all data points in the time series
            new_data.loc[df.sq_eps > df.sq_eps.mean(), feat] = None

        return self.__class__(strategy(new_data))

    # -- Interpolation and resampling --

    def split(self, value: int = 10, unit: str = "m") -> Iterator["Flight"]:
        """Splits Flights in several legs.

        By default, Flights are split if no value is given during 10 minutes.
        """
        for data in _split(self.data, value, unit):
            yield self.__class__(data)

    def resample(self, rule: str = "1s") -> "Flight":
        """Resamples a Flight at a one point per second rate. """
        data = (
            self.data.assign(start=self.start, stop=self.stop)
            .set_index("timestamp")
            .resample(rule)
            .interpolate()
            .reset_index()
            .fillna(method="pad")
        )
        return self.__class__(data)

    def at(self, time: timelike) -> pd.core.series.Series:
        index = to_datetime(time)
        return self.data.set_index("timestamp").loc[index]

    def between(self, before: timelike, after: time_or_delta) -> "Flight":
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)

        t: np.ndarray = np.stack(self.timestamp)
        index = np.where((before < t) & (t < after))
        return self.__class__(self.data.iloc[index])

    # -- Geometry operations --

    def extent(self) -> Tuple[float, float, float, float]:
        return (
            self.data.longitude.min() - .1,
            self.data.longitude.max() + .1,
            self.data.latitude.min() - .1,
            self.data.latitude.max() + .1,
        )

    def clip(
        self, shape: base.BaseGeometry
    ) -> Union[None, "Flight", Iterator["Flight"]]:

        linestring = LineString(list(self.airborne().xy_time))
        intersection = linestring.intersection(shape)

        if intersection.is_empty:
            return None

        if isinstance(intersection, LineString):
            times = list(
                datetime.fromtimestamp(t)
                for t in np.stack(intersection.coords)[:, 2]
            )
            return self.__class__(
                self.data[
                    (self.data.timestamp >= min(times))
                    & (self.data.timestamp <= max(times))
                ]
            )

        def _clip_generator():
            for segment in intersection:
                times = list(
                    datetime.fromtimestamp(t)
                    for t in np.stack(segment.coords)[:, 2]
                )
                yield self.__class__(
                    self.data[
                        (self.data.timestamp >= min(times))
                        & (self.data.timestamp <= max(times))
                    ]
                )

        return (leg for leg in _clip_generator())

    # -- Visualisation --

    def plot(self, ax: GeoAxesSubplot, **kwargs) -> None:
        if "projection" in ax.__dict__ and "transform" not in kwargs:
            from cartopy.crs import PlateCarree

            kwargs["transform"] = PlateCarree()
        if self.shape is not None:
            ax.plot(*self.shape.xy, **kwargs)
