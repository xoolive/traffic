# fmt: off

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import (TYPE_CHECKING, Callable, Dict, Generator, Iterable,
                    Iterator, List, Optional, Set, Tuple, Union, cast, overload)

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
from . import geodesy as geo
from .distance import (DistanceAirport, DistancePointTrajectory, closest_point,
                       guess_airport)
from .mixins import GeographyMixin, PointMixin, ShapelyMixin
from .time import time_or_delta, timelike, to_datetime

if TYPE_CHECKING:
    from .airspace import Airspace  # noqa: F401
    from .airport import Airport  # noqa: F401
    from .traffic import Traffic  # noqa: F401

# fmt: on

# fix https://github.com/xoolive/traffic/issues/12
# if pd.__version__ <= "0.24.1":
DatetimeTZBlock.interpolate = Block.interpolate


def _split(
    data: pd.DataFrame, value: Union[str, int], unit: Optional[str]
) -> Iterator[pd.DataFrame]:
    # This method helps splitting a flight into several.
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


class Flight(GeographyMixin, ShapelyMixin):
    """Flight is the most basic class associated to a trajectory.
    Flights are the building block of all processing methods, built on top of
    pandas DataFrame. The minimum set of required features are:

    - ``icao24``: the ICAO transponder ID of an aircraft;
    - ``callsign``: an identifier which may be associated with the
      registration of an aircraft, with its mission (VOR calibration,
      firefighting) or with a route (for a commercial aircraft);
    - ``timestamp``: timezone aware timestamps are preferable.
      Some methods may work with timezone naive timestamps but the behaviour
      is not guaranteed;
    - ``latitude``, ``longitude``: in degrees, WGS84 (EPSG:4326);
    - ``altitude``: in feet.

    .. note::
        The ``flight_id`` (identifier for a trajectory) may be used in place of
        a pair of (``icao24``, ``callsign``). More features may also be provided
        for further processing, e.g. ``groundspeed``, ``vertical_rate``,
        ``track``, ``heading``, ``IAS`` (indicated airspeed) or ``squawk``.

    **Abridged contents:**

        - properties:
          `callsign <#traffic.core.Flight.callsign>`_,
          `flight_id <#traffic.core.Flight.flight_id>`_,
          `icao24 <#traffic.core.Flight.icao24>`_,
          `number <#traffic.core.Flight.number>`_,
          `registration <#traffic.core.Flight.registration>`_,
          `start <#traffic.core.Flight.start>`_,
          `stop <#traffic.core.Flight.stop>`_,
          `typecode <#traffic.core.Flight.typecode>`_
        - time related methods:
          `after() <#traffic.core.Flight.after>`_,
          `at() <#traffic.core.Flight.at>`_,
          `at_ratio() <#traffic.core.Flight.at_ratio>`_,
          `before() <#traffic.core.Flight.before>`_,
          `between() <#traffic.core.Flight.between>`_,
          `first() <#traffic.core.Flight.first>`_,
          `last() <#traffic.core.Flight.last>`_
        - geometry related methods:
          `airborne() <#traffic.core.Flight.airborne>`_,
          `clip() <#traffic.core.Flight.clip>`_,
          `compute_wind() <#traffic.core.Flight.compute_wind>`_,
          `compute_xy() <#traffic.core.Flight.compute_xy>`_,
          `distance() <#traffic.core.Flight.distance>`_,
          `inside_bbox() <#traffic.core.Flight.inside_bbox>`_,
          `intersects() <#traffic.core.Flight.intersects>`_,
          `project_shape() <#traffic.core.Flight.project_shape>`_,
          `simplify() <#traffic.core.Flight.simplify>`_,
          `unwrap() <#traffic.core.Flight.unwrap>`_
        - filtering and resampling methods:
          `comet() <#traffic.core.Flight.comet>`_,
          `filter() <#traffic.core.Flight.filter>`_,
          `resample() <#traffic.core.Flight.resample>`_,
        - visualisation with altair:
          `encode() <#traffic.core.Flight.encode>`_,
          `geoencode() <#traffic.core.Flight.geoencode>`_
        - visualisation with leaflet: `layer() <#traffic.core.Flight.layer>`_
        - visualisation with Matplotlib:
          `plot() <#traffic.core.Flight.plot>`_,
          `plot_time() <#traffic.core.Flight.plot_time>`_

    .. tip::
        Sample flights are provided for testing purposes in module
        ``traffic.data.samples``

    """

    __slots__ = ("data",)

    # --- Special methods ---

    def __add__(self, other) -> "Traffic":
        """
        As Traffic is thought as a collection of Flights, the sum of two Flight
        objects returns a Traffic object
        """
        # keep import here to avoid recursion
        from .traffic import Traffic  # noqa: F811

        if other == 0:
            # useful for compatibility with sum() function
            return Traffic(self.data)

        return Traffic.from_flights([self, other])

    def __radd__(self, other) -> "Traffic":
        """
        As Traffic is thought as a collection of Flights, the sum of two Flight
        objects returns a Traffic object
        """
        return self + other

    def __len__(self) -> int:
        """Number of samples associated to a trajectory.

        The basic behaviour is to return the number of lines in the underlying
        DataFrame. However in some cases, as positions may be wrongly repeated
        in some database systems (e.g. OpenSky Impala shell), we take the
        `last_position` field into account for counting the number of unique
        detected positions.

        Note that when an aircraft is onground, `last_position` is a more
        relevant criterion than (`latitude`, `longitude`) since a grounded
        aircraft may be repeatedly emitting the same position.
        """

        if "last_position" in self.data.columns:
            return self.data.drop_duplicates("last_position").shape[0]
        else:
            return self.data.shape[0]

    def _info_html(self) -> str:
        title = f"<b>Flight {self.title}</b>"
        title += "<ul>"
        title += f"<li><b>aircraft:</b> {self.aircraft}</li>"
        if self.origin is not None:
            title += f"<li><b>from:</b> {self.origin} ({self.start})</li>"
        else:
            title += f"<li><b>from:</b> {self.start}</li>"
        if self.destination is not None:
            title += f"<li><b>to:</b> {self.destination} ({self.stop})</li>"
        else:
            title += f"<li><b>to:</b> {self.stop}</li>"
        title += "</ul>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def __repr__(self) -> str:
        output = f"Flight {self.title}"
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

    def filter_if(self, test: Callable[["Flight"], bool]) -> Optional["Flight"]:
        return self if test(self) else None

    # --- Iterators ---

    @property
    def timestamp(self) -> Iterator[pd.Timestamp]:
        yield from self.data.timestamp

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        data = self.data.query("longitude == longitude")
        yield from zip(data["longitude"], data["latitude"], data["altitude"])

    def coords4d(
        self, delta_t: bool = False
    ) -> Iterator[Tuple[float, float, float, float]]:
        data = self.data.query("longitude == longitude")
        if delta_t:
            time = (data.timestamp - data.timestamp.min()).dt.total_seconds()
        else:
            time = data["timestamp"]

        yield from zip(
            time, data["longitude"], data["latitude"], data["altitude"]
        )

    @property
    def xy_time(self) -> Iterator[Tuple[float, float, float]]:
        self_filtered = self.query("longitude == longitude")
        iterator = iter(zip(self_filtered.coords, self_filtered.timestamp))
        while True:
            next_ = next(iterator, None)
            if next_ is None:
                return
            coords, time = next_
            yield (coords[0], coords[1], time.to_pydatetime().timestamp())

    # --- Properties (and alike) ---

    @lru_cache()
    def min(self, feature: str):
        """Returns the minimum value of given feature."""
        return self.data[feature].min()

    @lru_cache()
    def max(self, feature: str):
        """Returns the maximum value of given feature."""
        return self.data[feature].max()

    @property
    def start(self) -> pd.Timestamp:
        """Returns the minimum value of timestamp."""
        return self.min("timestamp")

    @property
    def stop(self) -> pd.Timestamp:
        """Returns the maximum value of timestamp."""
        return self.max("timestamp")

    @property
    def duration(self) -> pd.Timedelta:
        """Returns the duration of the flight."""
        return self.stop - self.start

    def _get_unique(
        self, field: str, warn: bool = True
    ) -> Union[str, Set[str], None]:
        if field not in self.data.columns:
            return None
        tmp = self.data[field].unique()
        if len(tmp) == 1:
            return tmp[0]
        if warn:
            logging.warning(
                f"Several {field}s for one flight, consider splitting"
            )
        return set(tmp)

    @property
    def callsign(self) -> Union[str, Set[str], None]:
        """Returns the unique callsign value(s) associated to the Flight.

        A callsign is an identifier sent by an aircraft during its flight. It
        may be associated with the registration of an aircraft, its mission or
        with a route for a commercial aircraft.
        """
        callsign = self._get_unique("callsign")
        if callsign != callsign:
            raise ValueError("NaN appearing in callsign field")
        return callsign

    @property
    def number(self) -> Union[str, Set[str], None]:
        """Returns the unique number value(s) associated to the Flight.

        This field is reserved for the commercial number of the flight, prefixed
        by the two letter code of the airline.
        For instance, AFR292 is the callsign and AF292 is the flight number.

        Callsigns are often more complicated as they are designed to limit
        confusion on the radio: hence DLH02X can be the callsign associated
        to flight number LH1100.
        """
        return self._get_unique("number")

    @property
    def flight_id(self) -> Union[str, Set[str], None]:
        """Returns the unique flight_id value(s) of the DataFrame.

        Neither the icao24 (the aircraft) nor the callsign (the route) is a
        reliable way to identify trajectories. You can either use an external
        source of data to assign flight ids (for example DDR files by
        Eurocontrol, identifiers by FlightRadar24, etc.) or assign a flight_id
        by yourself (see ``Flight.assign_id(name: str)`` method).

        The ``Traffic.assign_id()`` method uses a heuristic based on the
        timestamps associated to callsign/icao24 pairs to automatically assign a
        ``flight_id`` and separate flights.

        """
        return self._get_unique("flight_id")

    @property
    def title(self) -> str:
        title = str(self.callsign)
        number = self.number
        flight_id = self.flight_id

        if number is not None:
            title += f" / {number}"

        if flight_id is not None:
            title += f" ({flight_id})"

        return title

    @property
    def origin(self) -> Union[str, Set[str], None]:
        """Returns the unique origin value(s),
        None if not available in the DataFrame.

        The origin airport is usually represented as a ICAO or a IATA code.

        The ICAO code of an airport is represented by 4 letters (e.g. EHAM for
        Amsterdam Schiphol International Airport) and the IATA code is
        represented by 3 letters and more familiar to the public (e.g. AMS for
        Amsterdam)

        """
        return self._get_unique("origin")

    @property
    def destination(self) -> Union[str, Set[str], None]:
        """Returns the unique destination value(s),
        None if not available in the DataFrame.

        The destination airport is usually represented as a ICAO or a IATA code.

        The ICAO code of an airport is represented by 4 letters (e.g. EHAM for
        Amsterdam Schiphol International Airport) and the IATA code is
        represented by 3 letters and more familiar to the public (e.g. AMS for
        Amsterdam)

        """
        return self._get_unique("destination")

    @property
    def squawk(self) -> Set[str]:
        """Returns all the unique squawk values in the trajectory.

        A squawk code is a four-digit number assigned by ATC and set on the
        transponder. Some squawk codes are reserved for specific situations and
        emergencies, e.g. 7700 for general emergency, 7600 for radio failure or
        7500 for hijacking.
        """
        return set(self.data.squawk.unique())

    @property
    def icao24(self) -> Union[str, Set[str], None]:
        """Returns the unique icao24 value(s) of the DataFrame.

        icao24 (ICAO 24-bit address) is a unique identifier associated to a
        transponder. These identifiers correlate to the aircraft registration.

        For example icao24 code 'ac82ec' is associated to 'N905NA'.
        """
        icao24 = self._get_unique("icao24")
        if icao24 != icao24:
            raise ValueError("NaN appearing in icao24 field")
        return icao24

    @property
    def registration(self) -> Optional[str]:
        from ..data import aircraft

        if not isinstance(self.icao24, str):
            return None
        res = aircraft[self.icao24]
        res = res.query("registration == registration and registration != ''")
        if res.shape[0] == 1:
            return res.iloc[0].registration
        return None

    @property
    def typecode(self) -> Optional[str]:
        from ..data import aircraft

        if not isinstance(self.icao24, str):
            return None
        res = aircraft[self.icao24]
        res = res.query("typecode == typecode and typecode != ''")
        if res.shape[0] == 1:
            return res.iloc[0].typecode
        return None

    @property
    def aircraft(self) -> Optional[str]:
        if not isinstance(self.icao24, str):
            return None

        res = str(self.icao24)
        registration = self.registration
        typecode = self.typecode

        if registration is not None:
            res += f" / {registration}"

        if typecode is not None:
            res += f" ({typecode})"

        return res

    # -- Time handling, splitting, interpolation and resampling --

    def first(self, **kwargs) -> "Flight":
        """Returns the first n days, hours, minutes or seconds of the Flight.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.first(minutes=10)
        """
        delta = timedelta(**kwargs)
        bound = self.start + delta  # noqa: F841 => used in the query
        # full call is necessary to keep @bound as a local variable
        return self.__class__(self.data.query("timestamp < @bound"))

    def last(self, **kwargs) -> "Flight":
        """Returns the last n days, hours, minutes or seconds of the Flight.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.last(minutes=10)
        """
        delta = timedelta(**kwargs)
        bound = self.stop - delta  # noqa: F841 => used in the query
        # full call is necessary to keep @bound as a local variable
        return self.__class__(self.data.query("timestamp > @bound"))

    def before(self, time: timelike) -> "Flight":
        """Returns the part of the trajectory flown before a given timestamp.

        - ``time`` can be passed as a string, an epoch, a Python datetime, or
          a Pandas timestamp.
        """
        return self.between(self.start, time)

    def after(self, time: timelike) -> "Flight":
        """Returns the part of the trajectory flown after a given timestamp.

        - ``time`` can be passed as a string, an epoch, a Python datetime, or
          a Pandas timestamp.
        """
        return self.between(time, self.stop)

    def between(self, start: timelike, stop: time_or_delta) -> "Flight":
        """Returns the part of the trajectory flown between start and stop.

        - ``start`` and ``stop`` can be passed as a string, an epoch, a Python
          datetime, or a Pandas timestamp.
        - ``stop`` can also be passed as a timedelta.

        """

        start = to_datetime(start)
        if isinstance(stop, timedelta):
            stop = start + stop
        else:
            stop = to_datetime(stop)

        # full call is necessary to keep @start and @stop as local variables
        # return self.query('@start < timestamp < @stop')  => not valid
        return self.__class__(self.data.query("@start < timestamp < @stop"))

    def at(self, time: Optional[timelike] = None) -> Optional[Position]:
        """Returns the position in the trajectory at a given timestamp.

        - ``time`` can be passed as a string, an epoch, a Python datetime, or
          a Pandas timestamp.

        - If no time is passed (default), the last know position is returned.
        - If no position is available at the given timestamp, None is returned.
          If you expect a position at any price, consider `Flight.resample
          <#traffic.core.Flight.resample>`_

        """

        if time is None:
            return Position(self.data.ffill().iloc[-1])

        index = to_datetime(time)
        df = self.data.set_index("timestamp")
        if index not in df.index:
            id_ = getattr(self, "flight_id", self.callsign)
            logging.warning(f"No index {index} for flight {id_}")
            return None
        return Position(df.loc[index])

    def at_ratio(self, ratio: float = 0.5) -> Optional[Position]:
        """Returns a position on the trajectory.

        This method is convenient to place a marker on the trajectory in
        visualisation output.

        - ``Flight.at_ratio(0)`` is the first point in the trajectory.
        - ``Flight.at_ratio(1)`` is the last point of the trajectory
          (equivalent to ``Flight.at()``)
        """
        return self.between(self.start, self.start + ratio * self.duration).at()

    @overload
    def split(self, value: int, unit: str) -> Iterator["Flight"]:
        ...

    @overload  # noqa: F811
    def split(self, value: str, unit: None = None) -> Iterator["Flight"]:
        ...

    def split(  # noqa: F811
        self, value: Union[int, str] = 10, unit: Optional[str] = None
    ) -> Iterator["Flight"]:
        """Iterates on legs of a Flight based on the distrution of timestamps.

        By default, the method stops a flight and yields a new one after a gap
        of 10 minutes without data.

        The length of the gap (here 10 minutes) can be expressed:

        - in the NumPy style: ``Flight.split(10, 'm')`` (see
          ``np.timedelta64``);
        - in the pandas style: ``Flight.split('10T')`` (see ``pd.Timedelta``)

        """
        if type(value) == int and unit is None:
            # default value is 10 m
            unit = "m"

        for data in _split(self.data, value, unit):
            yield self.__class__(data)

    def handle_last_position(self) -> "Flight":
        # The following is True for all data coming from the Impala shell.
        # The following is an attempt to fix #7
        # Note the fun/fast way to produce 1 or trigger NaN (division by zero)
        data = self.data.sort_values("timestamp")
        if "last_position" in self.data.columns:
            data = (
                data.assign(
                    _mark=lambda df: df.last_position
                    != df.shift(1).last_position
                ).assign(
                    latitude=lambda df: df.latitude * df._mark / df._mark,
                    longitude=lambda df: df.longitude * df._mark / df._mark,
                    altitude=lambda df: df.altitude * df._mark / df._mark,
                )
                # keeping last_position causes more problems (= Nan) than
                # anything. Safer to just remove it for now. Like it or not!
                .drop(columns=["_mark", "last_position"])
            )

        return self.__class__(data)

    def resample(self, rule: Union[str, int] = "1s") -> "Flight":
        """Resample the trajectory at a given frequency or number of points.

        If the rule is a string representing a pandas `time series frequency
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_
        is passed, then the data is resampled along the timestamp axis, then
        interpolated.

        If the rule is an integer, the trajectory is resampled to the given
        number of evenly distributed points per trajectory.
        """

        if isinstance(rule, str):
            data = (
                self.handle_last_position()
                .data.assign(start=self.start, stop=self.stop)
                .set_index("timestamp")
                .resample(rule)
                .first()  # better performance than min() for duplicate index
                .interpolate()
                .reset_index()
                .fillna(method="pad")
            )
        elif isinstance(rule, int):
            # ./site-packages/pandas/core/indexes/base.py:2820: FutureWarning:
            # Converting timezone-aware DatetimeArray to timezone-naive ndarray
            # with 'datetime64[ns]' dtype. In the future, this will return an
            # ndarray with 'object' dtype where each element is a
            # 'pandas.Timestamp' with the correct 'tz'.
            # To accept the future behavior, pass 'dtype=object'.
            # To keep the old behavior, pass 'dtype="datetime64[ns]"'.
            data = (
                self.handle_last_position()
                .assign(tz_naive=lambda d: d.timestamp.astype("datetime64[ns]"))
                .data.set_index("tz_naive")
                .asfreq((self.stop - self.start) / (rule - 1), method="nearest")
                .reset_index()
                .drop(columns="tz_naive")
            )
        else:
            raise TypeError("rule must be a str or an int")

        return self.__class__(data)

    def filter(
        self,
        strategy: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = lambda x: x.bfill().ffill(),
        **kwargs,
    ) -> "Flight":

        """Filters the trajectory given features with a median filter.

        The method applies a median filter on each feature of the DataFrame.
        A default kernel size is applied for a number of features (resp.
        latitude, longitude, altitude, track, groundspeed, IAS, TAS) but other
        kernel values may be passed as kwargs parameters.

        Filtered values are replaced by NaN values. A strategy may be applied to
        fill the Nan values, by default a forward/backward fill. Other
        strategies may be passed, for instance *do nothing*: ``lambda x: x``; or
        *interpolate*: ``lambda x: x.interpolate()``.

        .. note::
            This method if often more efficient when applied several times with
            different kernel values.

            >>> # this cascade of filters appears to work well on altitude
            >>> flight.resample().resample(altitude=53)
        """

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

    def comet(self, **kwargs) -> "Flight":
        """Computes a comet for a trajectory.

        The method uses the last position of a trajectory (method `at()
        <#traffic.core.Flight.at>`_) and uses the ``track`` (in degrees),
        ``groundspeed`` (in knots) and ``vertical_rate`` (in ft/min) values to
        interpolate the trajectory in a straight line.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.comet(minutes=10)
        >>> flight.before("2018-12-24 23:55").comet(minutes=10)  # Merry XMas!

        """

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

    # -- Air traffic management --

    def assign_id(
        self, name: str = "{self.callsign}_{idx:>03}", idx: int = 0
    ) -> "Flight":
        """Assigns a flight_id to a Flight.

        This method is more generally used by the corresponding Traffic and
        LazyTraffic methods but works fine on Flight as well.
        """
        return self.assign(flight_id=name.format(self=self, idx=idx))

    def airborne(self) -> "Flight":
        """Returns the airborne part of the Flight.

        The airborne part is determined by an ``onground`` flag or null values
        in the altitude column.
        """
        if "onground" in self.data.columns and self.data.onground.dtype == bool:
            return self.query("not onground and altitude == altitude")
        else:
            return self.query("altitude == altitude")

    def unwrap(
        self, features: Union[str, List[str]] = ["track", "heading"]
    ) -> "Flight":
        """Unwraps angles in the DataFrame.

        All features representing angles may be unwrapped (through Numpy) to
        avoid gaps between 359° and 1°.

        The method applies by default to features ``track`` and ``heading``.
        More or different features may be passed in parameter.
        """
        if isinstance(features, str):
            features = [features]

        result_dict = dict()
        for feature in features:
            if feature not in self.data.columns:
                continue
            result_dict[f"{feature}_unwrapped"] = np.degrees(
                np.unwrap(np.radians(self.data[feature]))
            )

        return self.assign(**result_dict)

    def compute_wind(self) -> "Flight":
        """Computes the wind triangle for each timestamp.

        This method requires ``groundspeed``, ``track``, true airspeed
        (``TAS``), and ``heading`` features. The groundspeed and the track angle
        are usually available in ADS-B messages; the heading and the true
        airspeed may be decoded in EHS messages.

        .. note::
            Check the `query_ehs() <#traffic.core.Flight.query_ehs>`_ method to
            find a way to enrich your flight with such features. Note that this
            data is not necessarily available depending on the location.
        """
        df = self.data
        return self.assign(
            wind_u=df.groundspeed * np.sin(np.radians(df.track))
            - df.TAS * np.sin(np.radians(df.heading)),
            wind_v=df.groundspeed * np.cos(np.radians(df.track))
            - df.TAS * np.cos(np.radians(df.heading)),
        )

    def closest_point(self, points: Union[List[PointMixin], PointMixin]):
        # TODO refactor/rethink return type and documentation
        if not isinstance(points, list):
            points = [points]
        return min(closest_point(self.data, point) for point in points)

    def guess_takeoff_airport(self) -> DistanceAirport:
        # TODO refactor/rethink return type and documentation
        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[0])

    def guess_landing_airport(self) -> DistanceAirport:
        # TODO refactor/rethink return type and documentation
        data = self.data.sort_values("timestamp")
        return guess_airport(data.iloc[-1])

    def guess_landing_runway(
        self, airport: Union[None, str, "Airport"] = None
    ) -> DistancePointTrajectory:
        # TODO refactor/rethink return type and documentation

        if airport is None:
            airport = self.guess_landing_airport().airport
        if isinstance(airport, str):
            from ..data import airports

            airport = airports[airport]

        all_runways: Dict[str, PointMixin] = dict()
        for p in airport.runways.list:  # type: ignore
            all_runways[p.name] = p

        subset = (
            self.airborne()
            .query("vertical_rate < 0")
            .last(minutes=10)
            .resample()
        )
        candidate = subset.closest_point(list(all_runways.values()))

        avg_track = subset.data.track.tail(10).mean()
        rwy_bearing = all_runways[candidate.name].bearing  # type: ignore

        if abs(avg_track - rwy_bearing) > 20:
            logging.warning(
                f"({self.flight_id}) Candidate runway "
                f"{candidate.name} is not consistent "
                f"with average track {avg_track}."
            )

        return candidate

    # -- Distances --

    @overload
    def distance(self, other: PointMixin) -> "Flight":
        ...

    @overload  # noqa: F811
    def distance(self, other: "Flight") -> pd.DataFrame:
        ...

    def distance(  # noqa: F811
        self, other: Union["Flight", PointMixin]
    ) -> Union["Flight", pd.DataFrame]:

        if isinstance(other, PointMixin):
            size = len(self)
            return self.assign(
                distance=geo.distance(
                    self.data.latitude.values,
                    self.data.longitude.values,
                    other.latitude * np.ones(size),
                    other.longitude * np.ones(size),
                )
                / 1852  # in nautical miles
            )

        start = max(self.airborne().start, other.airborne().start)
        stop = min(self.airborne().stop, other.airborne().stop)
        f1, f2 = (self.between(start, stop), other.between(start, stop))

        cols = ["timestamp", "latitude", "longitude", "altitude"]
        cols += ["icao24", "callsign"]
        if "flight_id" in f1.data.columns:
            cols.append("flight_id")
        table = f1.data[cols].merge(f2.data[cols], on="timestamp")

        return table.assign(
            lateral=geo.distance(
                table.latitude_x.values,
                table.longitude_x.values,
                table.latitude_y.values,
                table.longitude_y.values,
            )
            / 1852,  # in nautical miles
            vertical=(table.altitude_x - table.altitude_y).abs(),
        )

    def cumulative_distance(
        self, compute_groundspeed: bool = False
    ) -> "Flight":

        """ Enrich the structure with new ``cumdist`` column computed from
        latitude and longitude columns.

        The first ``cumdist`` value is 0, then distances are computed (in
        **nautical miles**) and summed between consecutive positions. The last
        value is the total length of the trajectory.

        When the ``compute_groundspeed`` flag is set to True, an additional
        ``compute_gs`` is also added. This value can be compared with the
        decoded ``groundspeed`` value in ADSB messages.

        """

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
            gs = d / delta_1.timestamp_1.dt.total_seconds() * (3600 / 1852)
            res = res.assign(compute_gs=np.pad(gs, (1, 0), "constant"))
        return res

    # -- Geometry operations --

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
    def point(self) -> Optional[Position]:
        return self.at()

    def simplify(
        self,
        tolerance: float,
        altitude: Optional[str] = None,
        z_factor: float = 3.048,
        return_mask: bool = False,
    ) -> Union[np.ndarray, "Flight"]:
        """Simplifies a trajectory with Douglas-Peucker algorithm.

        The method uses latitude and longitude, projects the trajectory to a
        conformal projection and applies the algorithm. If x and y features are
        already present in the DataFrame (after a call to `compute_xy()
        <#traffic.core.Flight.compute_xy>`_ for instance) then this projection
        is taken into account.

        - By default, a 2D version is called, unless you pass a column name for
          ``altitude``.
        - You may scale the z-axis for more relevance (``z_factor``). The
          default value works well in most situations.

        The method returns a Flight unless you specify ``return_mask=True``.
        """

        if "x" in self.data.columns and "y" in self.data.columns:
            kwargs = dict(x="x", y="y")
        else:
            kwargs = dict(lat="latitude", lon="longitude")

        mask = douglas_peucker(
            df=self.data,
            tolerance=tolerance,
            z=altitude,
            z_factor=z_factor,
            **kwargs,
        )

        if return_mask:
            return mask
        else:
            return self.__class__(self.data.loc[mask])

    def intersects(self, shape: Union["Airspace", base.BaseGeometry]) -> bool:
        # implemented and monkey-patched in airspace.py
        # given here for consistency in types
        ...

    def clip(
        self, shape: Union["Airspace", base.BaseGeometry]
    ) -> Optional["Flight"]:
        """Clips the trajectory to a given shape.

        For a shapely Geometry, the first time of entry and the last time of
        exit are first computed before returning the part of the trajectory
        between the two timestamps.

        Most of the time, aircraft do not repeatedly come out and in an
        airspace, but computation errors may sometimes give this impression.
        As a consequence, the clipped trajectory may have points outside the
        shape.

        .. warning::
            Altitudes are not taken into account.

        """

        linestring = LineString(list(self.airborne().xy_time))
        if not isinstance(shape, base.BaseGeometry):
            shape = shape.flatten()

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

    # -- OpenSky specific methods --

    def query_opensky(self) -> Optional["Flight"]:
        """Returns data from the same Flight as stored in OpenSky database.

        This may be useful if you write your own parser for data from a
        different channel. The method will use the ``callsign`` and ``icao24``
        attributes to build a request for current Flight in the OpenSky Network
        database.

        Returns None if no data is found.

        .. note::
            Read more about access to the OpenSky Network database `here
            <opensky_usage.html>`_
        """

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
        """Extends data with extra columns from EHS messages.

        By default, raw messages are requested from the OpenSky Network
        database.

        .. warning::
            Making a lot of small requests can be very inefficient and may look
            like a denial of service. If you get the raw messages using a
            different channel, you can provide the resulting dataframe as a
            parameter.

        The data parameter expect three columns: ``icao24``, ``rawmsg`` and
        ``mintime``, in conformance with the OpenSky API.

        .. note::
            Read more about access to the OpenSky Network database `here
            <opensky_usage.html>`_
        """
        from ..data import opensky, ModeS_Decoder

        if not isinstance(self.icao24, str):
            raise RuntimeError("Several icao24 for this flight")

        if not isinstance(self.callsign, str):
            raise RuntimeError("Several callsigns for this flight")

        def fail_warning():
            """Called when nothing can be added to data."""
            id_ = self.flight_id
            if id_ is None:
                id_ = self.callsign
            logging.warning(f"No data on Impala for flight {id_}.")
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
            )[["timestamp", "latitude", "longitude", "alt", "spd", "trk"]]
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

        if isinstance(self.origin, str):
            from ..data import airports

            airport = airports[self.origin]
            if airport is not None:
                decoder.acs.set_latlon(*airport.latlon)

        for _, line in progressbar(referenced_df.iterrows()):

            if line.alt < 5000 and line.latitude is not None:
                decoder.acs.set_latlon(line.latitude, line.longitude)

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

        aggregate = extended + self
        if "flight_id" in self.data.columns:
            aggregate.data.flight_id = self.flight_id

        # sometimes weird callsigns are decoded and should be discarded
        # so it seems better to filter on callsign rather than on icao24
        flight = aggregate[self.callsign]
        if flight is None:
            return failure()

        if self.number is not None:
            flight = flight.assign(number=self.number)
        if self.origin is not None:
            flight = flight.assign(origin=self.origin)
        if self.destination is not None:
            flight = flight.assign(destination=self.destination)

        return flight.sort_values("timestamp")

    # -- Visualisation --

    def plot(
        self, ax: GeoAxesSubplot, **kwargs
    ) -> List[Artist]:  # coverage: ignore
        """Plots the trajectory on a Matplotlib axis.

        The Flight supports Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        Example usage:

        >>> from traffic.drawing import Mercator
        >>> fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator())
        >>> flight.plot(ax, alpha=.5)

        .. note::
            See also `geoencode() <#traffic.core.Flight.geoencode>`_ for the
            altair equivalent.

        """

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            kwargs["transform"] = PlateCarree()
        if self.shape is not None:
            return ax.plot(*self.shape.xy, **kwargs)
        return []

    def encode(
        self, y: Union[str, List[str], alt.Y], **kwargs
    ) -> alt.Chart:  # coverage: ignore
        """Plots the given features according to time.

        The method ensures:

        - only non-NaN data are displayed (no gap in the plot);
        - the timestamp is naively converted to UTC if not localized.

        Example usage:

        >>> flight.encode('altitude')
        >>> # or with several comparable features
        >>> flight.encode(['groundspeed', 'IAS', 'TAS'])

        .. warning::
            No twin axes are available in altair/Vega charts.

        .. note::
            See also `plot_time() <#traffic.core.Flight.plot_time>`_ for the
            Matplotlib equivalent.
        """
        feature_list = ["timestamp"]
        alt_y: Optional[alt.Y] = None
        if "flight_id" in self.data.columns:
            feature_list.append("flight_id")
        if "callsign" in self.data.columns:
            feature_list.append("callsign")
        if "icao24" in self.data.columns:
            feature_list.append("icao24")
        if isinstance(y, alt.Y):
            alt_y = y
            y = y.shorthand
        if isinstance(y, str):
            feature_list.append(y)
            data = self.data[feature_list].query(f"{y} == {y}")
            default_encode = dict(
                x="timestamp:T",
                y=alt_y if alt_y is not None else alt.Y(y, title=y),
                color=alt.Color(
                    "flight_id"
                    if "flight_id" in data.columns
                    else (
                        "callsign" if "callsign" in data.columns else "icao24"
                    )
                ),
            )
        else:
            feature_list += y
            data = (
                self.data[feature_list]
                .melt("timestamp", y)
                .query("value == value")
            )
            default_encode = dict(x="timestamp:T", y="value", color="variable")

        return (
            alt.Chart(data)
            .mark_line(interpolate="monotone")
            .encode(**{**default_encode, **kwargs})
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
    ) -> None:  # coverage: ignore
        """Plots the given features according to time.

        The method ensures:

        - only non-NaN data are displayed (no gap in the plot);
        - the timestamp is naively converted to UTC if not localized.

        Example usage:

        >>> ax = plt.axes()
        >>> # most simple version
        >>> flight.plot(ax, 'altitude')
        >>> # or with several comparable features and twin axes
        >>> flight.plot(
        ...     ax, ['altitude', 'groundspeed, 'IAS', 'TAS'],
        ...     secondary_y=['altitude']
        ... )

        .. note::
            See also `encode() <#traffic.core.Flight.encode>`_ for the altair
            equivalent.

        """
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
