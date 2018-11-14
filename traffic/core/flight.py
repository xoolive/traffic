# fmt: off

import logging
import re
from datetime import datetime, timedelta
from typing import (TYPE_CHECKING, Callable, Generator, Iterable, Iterator,
                    List, NamedTuple, Optional, Set, Tuple, Union, cast)

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes

import pandas as pd
import scipy.signal
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxesSubplot
from shapely.geometry import LineString, base
from tqdm.autonotebook import tqdm

from . import geodesy as geo
from ..core.time import time_or_delta, timelike, to_datetime
from .distance import (DistanceAirport, DistancePointTrajectory, closest_point,
                       guess_airport)
from .mixins import DataFrameMixin, GeographyMixin, PointMixin, ShapelyMixin

if TYPE_CHECKING:
    from .airspace import Airspace  # noqa: F401

# fmt: on


def _split(data: pd.DataFrame, value, unit) -> Iterator[pd.DataFrame]:
    diff = data.timestamp.diff().values
    if diff.max() > np.timedelta64(value, unit):
        yield from _split(data.iloc[: diff.argmax()], value, unit)
        yield from _split(data.iloc[diff.argmax() :], value, unit)  # noqa
    else:
        yield data


class Position(PointMixin, pd.core.series.Series):
    pass


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

    @property
    def start(self) -> pd.Timestamp:
        """Returns the minimum timestamp value of the DataFrame."""
        return min(self.timestamp)

    @property
    def stop(self) -> pd.Timestamp:
        """Returns the maximum timestamp value of the DataFrame."""
        return max(self.timestamp)

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

    def query_ehs(self) -> "Flight":
        """Extend data with extra columns from EHS."""
        from ..data import opensky, ModeS_Decoder

        if not isinstance(self.icao24, str):
            raise RuntimeError("Several icao24 for this flight")

        def failure():
            """Called when nothing can be added to data."""
            id_ = self.flight_id
            if id_ is None:
                id_ = self.callsign
            logging.warn(f"No data on Impala for flight {id_}.")
            return self

        df = opensky.extended(self.start, self.stop, icao24=self.icao24)

        if df is None:
            return failure()

        timestamped_df = df.sort_values("mintime").assign(
            timestamp=lambda df: df.mintime.apply(
                lambda x: x.replace(microsecond=0)
            )
        )

        referenced_df = (
            timestamped_df.merge(self.data, on="timestamp", how="outer")
            .sort_values("timestamp")
            .rename(
                columns=dict(altitude_y="alt", groundspeed="spd", track="trk")
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
        for _, line in tqdm(
            referenced_df.iterrows(),
            total=referenced_df.shape[0],
            desc=f"{identifier}:",
            leave=False,
        ):

            decoder.process(
                line.timestamp,
                line.rawmsg,
                spd=line.spd,
                trk=line.trk,
                alt=line.alt,
            )

        if decoder.traffic is None:
            return failure()

        t = decoder.traffic[self.icao24] + self
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
            .query("vertical_rate < 0")
            .last(minutes=10)
            .resample()
        )
        candidate = subset.closest_point(all_runways)

        avg_track = subset.data.track.tail(10).mean()
        # TODO compute rwy track in the data module
        rwy_track = 10 * int(next(re.finditer("\d+", candidate.name)).group())

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
            time = (data["timestamp"] - data["timestamp"].min()).apply(
                lambda x: x.total_seconds()
            )
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
        return self.__class__(self.data[self.data["altitude"].notnull()])

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

    def distance(self, other: "Flight") -> pd.DataFrame:

        start = max(self.airborne().start, other.airborne().start)
        stop = min(self.airborne().stop, other.airborne().stop)
        f1, f2 = (
            self.between(start, stop).resample("1s"),
            other.between(start, stop).resample("1s"),
        )

        cols = ["timestamp", "latitude", "longitude", "altitude"]
        table = f1.data[cols].merge(f2.data[cols], on="timestamp")

        return table.assign(
            timestamp=lambda df: df.timestamp.dt.tz_localize(
                datetime.now().astimezone().tzinfo
            ).dt.tz_convert("utc"),
            d_horz=geo.distance(
                table.latitude_x.values,
                table.longitude_x.values,
                table.latitude_y.values,
                table.longitude_y.values,
            )
            / 1852,
            d_vert=(table.altitude_x - table.altitude_y).abs(),
        )

    # -- Interpolation and resampling --

    def split(self, value: int = 10, unit: str = "m") -> Iterator["Flight"]:
        """Splits Flights in several legs.

        By default, Flights are split if no value is given during 10 minutes.
        """
        for data in _split(self.data, value, unit):
            yield self.__class__(data)

    def resample(self, rule: Union[str, int] = "1s") -> "Flight":
        """Resamples a Flight at a one point per second rate."""

        if isinstance(rule, str):
            data = (
                self.data.assign(start=self.start, stop=self.stop)
                .set_index("timestamp")
                .resample(rule)
                .min()
                .interpolate()
                .reset_index()
                .fillna(method="pad")
            )
        elif isinstance(rule, int):
            data = (
                self.data.set_index("timestamp")
                .asfreq((self.stop - self.start) / (rule - 1), method="nearest")
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
                datetime.fromtimestamp(t)
                for t in np.stack(intersection.coords)[:, 2]
            )
            return self.between(min(times), max(times))

        def _clip_generator() -> Generator[
            Tuple[pd.Timestamp, pd.Timestamp], None, None
        ]:
            for segment in intersection:
                times = list(
                    datetime.fromtimestamp(t)
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
