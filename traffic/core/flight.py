# fmt: off

import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import (TYPE_CHECKING, Callable, Generator, Iterable, Iterator,
                    List, NamedTuple, Optional, Set, Tuple, Type, TypeVar,
                    Union, cast, overload)

import altair as alt
import numpy as np
import pandas as pd
import scipy.signal
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes
from pandas.core.internals import Block, DatetimeTZBlock
from shapely.geometry import LineString, base
from tqdm.autonotebook import tqdm

from ..algorithms.douglas_peucker import douglas_peucker
from ..core.time import time_or_delta, timelike, to_datetime
from . import geodesy as geo
from .distance import (DistanceAirport, DistancePointTrajectory, closest_point,
                       guess_airport)
from .mixins import GeographyMixin, PointMixin, ShapelyMixin

if TYPE_CHECKING:
    from .airspace import Airspace  # noqa: F401

# fmt: on

# fix https://github.com/xoolive/traffic/issues/12
# if pd.__version__ <= "0.24.1":
DatetimeTZBlock.interpolate = Block.interpolate


def _split(
    data: pd.DataFrame, value: Union[str, int], unit: Optional[str]
) -> Iterator[pd.DataFrame]:
    if data.shape[0] < 2:
        return
    diff = data.timestamp.diff().values
    if unit is None:
        delta = pd.Timedelta(value).to_timedelta64()
    else:
        delta = np.timedelta64(value, unit)
    if diff.max() > delta:
        yield from _split(data.iloc[: diff.argmax()], value, unit)
        yield from _split(data.iloc[diff.argmax() :], value, unit)  # noqa
    else:
        yield data


class Position(PointMixin, pd.core.series.Series):
    pass


# Typing for Douglas-Peucker
# It is comfortable to be able to return a Flight or a mask
Mask = TypeVar("Mask", "Flight", np.ndarray)


class Flight(GeographyMixin, ShapelyMixin):
    """Flight is the basic class associated to an aircraft itinerary.

    A Flight is supposed to start at takeoff and end after landing, taxiing and
    parking.

    If the current structure seems to contain many flights, warnings may be
    raised.
    """

    __slots__ = ("data",)

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

    def __repr__(self) -> str:
        output = f"Flight {self.callsign}"
        if self.number is not None:
            output += f" / {self.number}"
        if self.flight_id is not None:
            output += f" ({self.flight_id})"
        output += f"\naircraft: {self.aircraft}"
        if self.origin is not None:
            output += f"\norigin: {self.origin} ({self.start})"
        else:
            output += f"\norigin: {self.start}"
        if self.destination is not None:
            output += f"\ndestination: {self.destination} ({self.stop})"
        else:
            output += f"\ndestination: {self.stop}"
        return output

    @property
    def timestamp(self) -> Iterator[pd.Timestamp]:
        """Iterates the timestamp column of the DataFrame."""
        yield from self.data.timestamp

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def start(self) -> pd.Timestamp:
        """Returns the minimum timestamp value of the DataFrame."""
        start = self.data.timestamp.min()
        self.data = self.data.assign(start=start)
        return start

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def stop(self) -> pd.Timestamp:
        stop = self.data.timestamp.max()
        self.data = self.data.assign(stop=stop)
        return stop

    @lru_cache()
    def min(self, feature: str):
        return self.data[feature].min()

    @lru_cache()
    def max(self, feature: str):
        return self.data[feature].max()

    @property
    def callsign(self) -> Union[str, Set[str], None]:
        """Returns the unique callsign value(s) of the DataFrame."""
        if "callsign" not in self.data.columns:
            return None
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
        if all(self.data.number.isna()):
            return None
        tmp = set(self.data.number)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several numbers for one flight, consider splitting")
        return tmp

    @property
    def icao24(self) -> Union[str, Set[str], None]:
        """Returns the unique icao24 value(s) of the DataFrame.

        icao24 is a unique identifier associated to a transponder.
        """
        if "icao24" not in self.data.columns:
            return None
        tmp = set(self.data.icao24)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several icao24 for one flight, consider splitting")
        return tmp

    @property
    def flight_id(self) -> Optional[str]:
        """Returns the unique flight_id value(s) of the DataFrame.

        If you know how to split flights, you may want to append such a column
        in the DataFrame.
        """
        if "flight_id" not in self.data.columns:
            return None
        tmp = set(self.data.flight_id)
        if len(tmp) != 1:
            logging.warn("Several ids for one flight, consider splitting")
        return tmp.pop()

    @property
    def squawk(self) -> Set[str]:
        """Returns all the unique squawk values in the trajectory."""
        return set(self.data.squawk.ffill().bfill())

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

    def query_opensky(self) -> Optional["Flight"]:
        """Return the equivalent Flight from OpenSky History."""
        from ..data import opensky

        query_params = {
            "start": self.start,
            "stop": self.stop,
            "callsign": self.callsign,
            "icao24": self.icao24,
        }
        return opensky.history(**query_params)  # type: ignore

    def query_ehs(
        self,
        data: Optional[pd.DataFrame] = None,
        failure_mode: str = "warning",
        progressbar: Optional[Callable[[Iterable], Iterable]] = None,
    ) -> "Flight":
        """Extend data with extra columns from EHS messages.

        By default, raw messages are requested from the OpenSky Impala server.

        Making a lot of small requests can be very inefficient and may look
        like a denial of service. If you get the raw messages using a different
        channel, you can provide the resulting dataframe as a parameter.

        The data parameter expect three colmuns: icao24, rawmsg and mintime, in
        conformance with the OpenSky API.
        """
        from ..data import opensky, ModeS_Decoder

        if not isinstance(self.icao24, str):
            raise RuntimeError("Several icao24 for this flight")

        def fail_warning():
            """Called when nothing can be added to data."""
            id_ = self.flight_id
            if id_ is None:
                id_ = self.callsign
            logging.warn(f"No data on Impala for flight {id_}.")
            return self

        def fail_silent():
            return self

        failure_dict = dict(warning=fail_warning, silent=fail_silent)
        failure = failure_dict[failure_mode]

        if data is None:
            df = opensky.extended(self.start, self.stop, icao24=self.icao24)
        else:
            df = data.query("icao24 == @self.icao24").sort_values("mintime")

        if df is None:
            return failure()

        timestamped_df = df.sort_values("mintime").assign(
            timestamp=lambda df: df.mintime.dt.round("s")
        )

        referenced_df = (
            timestamped_df.merge(self.data, on="timestamp", how="outer")
            .sort_values("timestamp")
            .rename(
                columns=dict(
                    altitude="alt",
                    altitude_y="alt",
                    groundspeed="spd",
                    track="trk",
                )
            )[["timestamp", "alt", "spd", "trk"]]
            .ffill()
            .drop_duplicates()  # bugfix! NEVER ERASE THAT LINE!
            .merge(
                timestamped_df[["timestamp", "icao24", "rawmsg"]],
                on="timestamp",
                how="right",
            )
        )

        identifier = (
            self.flight_id if self.flight_id is not None else self.callsign
        )
        # who cares about default lat0, lon0 with EHS
        decoder = ModeS_Decoder((0, 0))

        if progressbar is None:
            progressbar = lambda x: tqdm(  # noqa: E731
                x,
                total=referenced_df.shape[0],
                desc=f"{identifier}:",
                leave=False,
            )

        for _, line in progressbar(referenced_df.iterrows()):

            decoder.process(
                line.timestamp,
                line.rawmsg,
                spd=line.spd,
                trk=line.trk,
                alt=line.alt,
            )

        if decoder.traffic is None:
            return failure()

        extended = decoder.traffic[self.icao24]
        if extended is None:
            return failure()

        # fix for https://stackoverflow.com/q/53657210/1595335
        if "last_position" in self.data.columns:
            extended = extended.assign(last_position=pd.NaT)
        if "start" in self.data.columns:
            extended = extended.assign(start=pd.NaT)
        if "stop" in self.data.columns:
            extended = extended.assign(stop=pd.NaT)

        t = extended + self
        if "flight_id" in self.data.columns:
            t.data.flight_id = self.flight_id

        # sometimes weird callsigns are decoded and should be discarded
        # so it seems better to filter on callsign rather than on icao24
        flight = t[self.callsign]
        if flight is None:
            return failure()

        return flight.sort_values("timestamp")

    def compute_wind(self) -> "Flight":
        df = self.data
        return self.assign(
            wind_u=df.groundspeed * np.sin(np.radians(df.track))
            - df.TAS * np.sin(np.radians(df.heading)),
            wind_v=df.groundspeed * np.cos(np.radians(df.track))
            - df.TAS * np.cos(np.radians(df.heading)),
        )

    def guess_takeoff_airport(self) -> DistanceAirport:
        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[0])

    def guess_landing_airport(self) -> DistanceAirport:
        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[-1])

    def guess_landing_runway(
        self, airport: Union[None, str, PointMixin] = None
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
            .query("vertical_rate < 0")
            .last(minutes=10)
            .resample()
        )
        candidate = subset.closest_point(all_runways)

        avg_track = subset.data.track.tail(10).mean()
        # TODO compute rwy track in the data module
        rwy_track = 10 * int(  # noqa: W605
            next(re.finditer("\d+", candidate.name)).group()
        )

        if abs(avg_track - rwy_track) > 20:
            logging.warn(
                f"({self.flight_id}) Candidate runway "
                f"{candidate.name} is not consistent "
                f"with average track {avg_track}."
            )

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
            # TODO return Aircraft and redirect this to __repr__
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
        if delta_t:
            time = (
                data["timestamp"] - data["timestamp"].min()
            ).dt.total_seconds()
        else:
            time = data["timestamp"]

        yield from zip(
            time, data["longitude"], data["latitude"], data["altitude"]
        )

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        """Iterates on longitudes, latitudes and altitudes.

        """
        data = self.data[self.data.longitude.notnull()]
        yield from zip(data["longitude"], data["latitude"], data["altitude"])

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

    @property
    def point(self) -> Optional[PointMixin]:
        positions = self.data.query("latitude == latitude")
        if len(positions) > 0:
            x = positions.iloc[-1]
            point = PointMixin()
            point.latitude = x.latitude
            point.longitude = x.longitude
            point.altitude = x.altitude
            point.timestamp = x.timestamp
            return point
        return None

    def airborne(self) -> "Flight":
        """Returns the airborne part of the Flight.

        The airborne part is determined by null values on the altitude column.
        """
        return self.query("altitude == altitude")

    def first(self, **kwargs) -> "Flight":
        delta = timedelta(**kwargs)
        bound = (  # noqa: F841 => used in the query
            cast(pd.Timestamp, self.start) + delta
        )
        return self.__class__(self.data.query("timestamp < @bound"))

    def last(self, **kwargs) -> "Flight":
        delta = timedelta(**kwargs)
        bound = (  # noqa: F841 => used in the query
            cast(pd.Timestamp, self.stop) - delta
        )
        return self.__class__(self.data.query("timestamp > @bound"))

    def filter(
        self,
        strategy: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = lambda x: x.bfill().ffill(),
        **kwargs,
    ) -> "Flight":

        ks_dict = {
            "altitude": 17,
            "track": 5,
            "groundspeed": 5,
            "longitude": 15,
            "latitude": 15,
            "IAS": 5,
            "TAS": 5,
            **kwargs,
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
                    "sigma": np.sqrt(filt(sq_eps, kernel_size)),
                },
                index=df.index,
            )

        new_data = self.data.sort_values(by="timestamp").copy()

        if len(kwargs) == 0:
            features = [
                cast(str, feature)
                for feature in self.data.columns
                if self.data[feature].dtype
                in [np.float32, np.float64, np.int32, np.int64]
            ]
        else:
            features = list(kwargs.keys())

        kernels_size = [0 for _ in features]
        for idx, feature in enumerate(features):
            kernels_size[idx] = ks_dict.get(feature, 17)

        for feat, ks in zip(features, kernels_size):

            # Prepare each flight for the filtering
            df = cascaded_filters(new_data[["timestamp", feat]], feat, ks)

            # Decision to accept/reject for all data points in the time series
            new_data.loc[df.sq_eps > df.sigma, feat] = None

        return self.__class__(strategy(new_data))

    @overload
    def distance(self, other: PointMixin) -> "Flight":
        ...

    @overload  # noqa: F811
    def distance(self, other: "Flight") -> pd.DataFrame:
        ...

    def distance(self, other):  # noqa: F811

        if isinstance(other, PointMixin):
            size = len(self)
            return self.assign(
                distance=geo.distance(
                    self.data.latitude.values,
                    self.data.longitude.values,
                    other.lat * np.ones(size),
                    other.lon * np.ones(size),
                )
                / 1852
            )

        start = max(self.airborne().start, other.airborne().start)
        stop = min(self.airborne().stop, other.airborne().stop)
        f1, f2 = (self.between(start, stop), other.between(start, stop))

        cols = ["timestamp", "latitude", "longitude", "altitude"]
        if "flight_id" in f1.data.columns:
            cols.append("flight_id")
        else:
            cols += ["icao24", "callsign"]
        table = f1.data[cols].merge(f2.data[cols], on="timestamp")

        return table.assign(
            lateral=geo.distance(
                table.latitude_x.values,
                table.longitude_x.values,
                table.latitude_y.values,
                table.longitude_y.values,
            )
            / 1852,
            vertical=(table.altitude_x - table.altitude_y).abs(),
        )

    def cumulative_distance(
        self, compute_groundspeed: bool = False
    ) -> "Flight":

        coords = self.data[["timestamp", "latitude", "longitude"]]
        delta = pd.concat([coords, coords.add_suffix("_1").diff()], axis=1)
        delta_1 = delta.iloc[1:]
        d = geo.distance(
            delta_1.latitude.values,
            delta_1.longitude.values,
            (delta_1.latitude + delta_1.latitude_1).values,
            (delta_1.longitude + delta_1.longitude_1).values,
        )

        res = self.assign(cumdist=np.pad(d.cumsum() / 1852, (1, 0), "constant"))

        if compute_groundspeed:
            res = res.assign(
                compute_gs=np.pad(
                    d / delta.timestamp_1.dt.total_seconds() * 3600 / 1852,
                    (1, 0),
                    "constant",
                )
            )
        return res

    # -- Interpolation and resampling --

    @overload
    def split(self, value: int, unit: str) -> Iterator["Flight"]:
        ...

    @overload  # noqa: F811
    def split(self, value: str, unit: None = None) -> Iterator["Flight"]:
        ...

    def split(self, value=10, unit=None):  # noqa: F811
        """Splits Flights in several legs.

        By default, Flights are split if no value is given during 10 minutes.
        """
        if type(value) == int and unit is None:
            # default value is 10 m
            unit = "m"

        for data in _split(self.data, value, unit):
            yield self.__class__(data)

    def _handle_last_position(self) -> "Flight":
        # The following is True for all data coming from the Impala shell.
        # The following is an attempt to fix #7
        # Note the fun/fast way to produce 1 or trigger NaN (division by zero)
        data = self.data.sort_values("timestamp")
        if "last_position" in self.data.columns:
            data = (
                data.assign(
                    _mark=lambda df: df.last_position
                    != df.shift(1).last_position
                )
                .assign(
                    latitude=lambda df: df.latitude * df._mark / df._mark,
                    longitude=lambda df: df.longitude * df._mark / df._mark,
                    altitude=lambda df: df.altitude * df._mark / df._mark,
                )
                .drop(columns="_mark")
            )

        return self.__class__(data)

    def resample(self, rule: Union[str, int] = "1s") -> "Flight":
        """Resamples a Flight at a one point per second rate."""

        if isinstance(rule, str):
            data = (
                self._handle_last_position()
                .data.assign(start=self.start, stop=self.stop)
                .set_index("timestamp")
                .resample(rule)
                .first()  # better performance than min() for duplicate index
                .interpolate()
                .reset_index()
                .fillna(method="pad")
            )
        elif isinstance(rule, int):
            data = (
                self._handle_last_position()
                .data.set_index("timestamp")
                .asfreq(
                    (self.stop - self.start) / (rule - 1),  # type: ignore
                    method="nearest",
                )
                .reset_index()
            )
        else:
            raise TypeError("rule must be a str or an int")

        return self.__class__(data)

    def comet(self, **kwargs) -> "Flight":

        last_line = self.at()
        if last_line is None:
            raise ValueError("Unknown data for this flight")
        window = self.last(seconds=20)
        delta = timedelta(**kwargs)

        new_gs = window.data.groundspeed.mean()
        new_vr = window.data.vertical_rate.mean()

        new_lat, new_lon, _ = geo.destination(
            last_line.latitude,
            last_line.longitude,
            last_line.track,
            new_gs * delta.total_seconds() * 1852 / 3600,
        )

        new_alt = last_line.altitude + new_vr * delta.total_seconds() / 60

        return Flight(
            pd.DataFrame.from_records(
                [
                    last_line,
                    pd.Series(
                        {
                            "timestamp": last_line.timestamp + delta,
                            "latitude": new_lat,
                            "longitude": new_lon,
                            "altitude": new_alt,
                            "groundspeed": new_gs,
                            "vertical_rate": new_vr,
                        }
                    ),
                ]
            ).ffill()
        )

    def at(self, time: Optional[timelike] = None) -> Optional[Position]:

        if time is None:
            return Position(self.data.ffill().iloc[-1])

        index = to_datetime(time)
        df = self.data.set_index("timestamp")
        if index not in df.index:
            id_ = getattr(self, "flight_id", self.callsign)
            logging.warn(f"No index {index} for flight {id_}")
            return None
        return Position(df.loc[index])

    def before(self, ts: timelike) -> "Flight":
        return self.between(self.start, ts)

    def after(self, ts: timelike) -> "Flight":
        return self.between(ts, self.stop)

    def between(self, start: timelike, stop: time_or_delta) -> "Flight":
        start = to_datetime(start)
        if isinstance(stop, timedelta):
            stop = start + stop
        else:
            stop = to_datetime(stop)

        # full call is necessary to keep @start and @stop as local variables
        # return self.query('@start < timestamp < @stop')  => not valid
        return self.__class__(self.data.query("@start < timestamp < @stop"))

    # -- Geometry operations --

    def simplify(
        self,
        tolerance: float,
        altitude: Optional[str] = None,
        z_factor: float = 3.048,
        return_type: Type[Mask] = Type["Flight"],
    ) -> Mask:
        """Simplifies a trajectory with Douglas-Peucker algorithm.

        The method uses latitude and longitude, projects the trajectory to a
        conformal projection and applies the algorithm. By default, a 2D version
        is called, unless you pass a column name for altitude (z parameter). You
        may scale the z-axis for more relevance (z_factor); the default value
        works well in most situations.

        The method returns a Flight unless you specify a np.ndarray[bool] as
        return_type for getting a mask.
        """
        # returns a mask
        mask = douglas_peucker(
            df=self.data,
            tolerance=tolerance,
            lat="latitude",
            lon="longitude",
            z=altitude,
            z_factor=z_factor,
        )
        if return_type == Type["Flight"]:
            return self.__class__(self.data.loc[mask])
        else:
            return mask

    def extent(self) -> Tuple[float, float, float, float]:
        return (
            self.data.longitude.min() - 0.1,
            self.data.longitude.max() + 0.1,
            self.data.latitude.min() - 0.1,
            self.data.latitude.max() + 0.1,
        )

    def intersects(self, airspace: "Airspace") -> bool:
        # implemented and monkey-patched in airspace.py
        # given here for consistency in types
        raise NotImplementedError

    def clip(self, shape: base.BaseGeometry) -> Optional["Flight"]:

        linestring = LineString(list(self.airborne().xy_time))
        intersection = linestring.intersection(shape)

        if intersection.is_empty:
            return None

        if isinstance(intersection, LineString):
            times = list(
                datetime.fromtimestamp(t, timezone.utc)
                for t in np.stack(intersection.coords)[:, 2]
            )
            return self.between(min(times), max(times))

        def _clip_generator() -> Generator[
            Tuple[pd.Timestamp, pd.Timestamp], None, None
        ]:
            for segment in intersection:
                times = list(
                    datetime.fromtimestamp(t, timezone.utc)
                    for t in np.stack(segment.coords)[:, 2]
                )
                yield min(times), max(times)

        times = list(_clip_generator())
        return self.between(min(t for t, _ in times), max(t for _, t in times))

    # -- Visualisation --

    def plot(self, ax: GeoAxesSubplot, **kwargs) -> List[Artist]:

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            kwargs["transform"] = PlateCarree()
        if self.shape is not None:
            return ax.plot(*self.shape.xy, **kwargs)
        return []

    def chart(
        self, feature_name: Union[str, List[str]], encode_dict: dict = dict()
    ) -> alt.Chart:
        feature_list = ["timestamp"]
        if "flight_id" in self.data.columns:
            feature_list.append("flight_id")
        if "callsign" in self.data.columns:
            feature_list.append("callsign")
        if "icao24" in self.data.columns:
            feature_list.append("icao24")
        if isinstance(feature_name, str):
            feature_list.append(feature_name)
            data = self.data.query(f"{feature_name} == {feature_name}")
            default_encode = dict(
                x="timestamp:T",
                y=alt.Y(feature_name, title=feature_name),
                color=alt.Color(
                    "flight_id"
                    if "flight_id" in data.columns
                    else (
                        "callsign" if "callsign" in data.columns else "icao24"
                    )
                ),
            )
        else:
            feature_list += feature_name
            data = self.data.melt("timestamp", feature_name).query(
                "value == value"
            )
            default_encode = dict(x="timestamp:T", y="value", color="variable")

        return (
            alt.Chart(data)
            .mark_line(interpolate="bundle")
            .encode(**{**default_encode, **encode_dict})
            .transform_timeunit(
                timestamp="utcyearmonthdatehoursminutesseconds(timestamp)"
            )
        )

    def plot_time(
        self,
        ax: Axes,
        y: Union[str, List[str]],
        secondary_y: Union[None, str, List[str]] = None,
        **kwargs,
    ) -> None:
        if isinstance(y, str):
            y = [y]
        if isinstance(secondary_y, str):
            secondary_y = [secondary_y]
        if secondary_y is None:
            secondary_y = []

        localized = self.data.timestamp.dt.tz is not None
        for column in y:
            kw = {
                **kwargs,
                **dict(
                    y=column,
                    secondary_y=column if column in secondary_y else "",
                ),
            }
            subtab = self.data.query(f"{column} == {column}")

            if localized:
                (
                    subtab.assign(
                        timestamp=lambda df: df.timestamp.dt.tz_convert("utc")
                    ).plot(ax=ax, x="timestamp", **kw)
                )
            else:
                (
                    subtab.assign(
                        timestamp=lambda df: df.timestamp.dt.tz_localize(
                            datetime.now().astimezone().tzinfo
                        ).dt.tz_convert("utc")
                    ).plot(ax=ax, x="timestamp", **kw)
                )
