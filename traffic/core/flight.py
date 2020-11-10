# fmt: off

import logging
import warnings
from datetime import datetime, timedelta, timezone
from operator import attrgetter
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator,
    List, Optional, Set, Tuple, Union, cast, overload
)

import altair as alt
import numpy as np
import pandas as pd
import pyproj
import scipy.signal
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes
from pandas.core.internals import DatetimeTZBlock
from shapely.geometry import LineString, MultiPoint, Point, Polygon, base
from shapely.ops import transform

from ..algorithms.douglas_peucker import douglas_peucker
from ..algorithms.navigation import NavigationFeatures
from ..algorithms.phases import FuzzyLogic
from ..drawing.markers import aircraft as aircraft_marker
from ..drawing.markers import rotate_marker
from . import geodesy as geo
from .iterator import FlightIterator, flight_iterator
from .mixins import GeographyMixin, HBoxMixin, PointMixin, ShapelyMixin
from .structure import Airport  # noqa: F401
from .time import deltalike, time_or_delta, timelike, to_datetime, to_timedelta

if TYPE_CHECKING:
    from ..data.adsb.raw_data import RawData  # noqa: F401
    from .airspace import Airspace  # noqa: F401
    from .lazy import LazyTraffic  # noqa: F401
    from .traffic import Traffic  # noqa: F401

# fmt: on


def _tz_interpolate(data, *args, **kwargs):
    return data.astype(int).interpolate(*args, **kwargs).astype(data.dtype)


DatetimeTZBlock.interpolate = _tz_interpolate


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
    # There seems to be a change with numpy >= 1.18
    # max() now may return NaN, therefore the following fix
    max_ = np.nanmax(diff)
    if max_ > delta:
        # np.nanargmax seems bugged with timestamps
        argmax = np.where(diff == max_)[0][0]
        yield from _split(data.iloc[:argmax], value, unit)
        yield from _split(data.iloc[argmax:], value, unit)  # noqa
    else:
        yield data


# flake B008
attrgetter_duration = attrgetter("duration")

# flake B006
default_angle_features = ["track", "heading"]


class Position(PointMixin, pd.core.series.Series):
    def plot(
        self, ax: Axes, text_kw=None, shift=None, **kwargs
    ) -> List[Artist]:  # coverage: ignore

        visualdict = dict(s=300)
        if hasattr(self, "track"):
            visualdict["marker"] = rotate_marker(aircraft_marker, self.track)

        if text_kw is None:
            text_kw = dict()
        else:
            # since we may modify it, let's make a copy
            text_kw = {**text_kw}

        if "s" not in text_kw and hasattr(self, "callsign"):
            text_kw["s"] = self.callsign

        return super().plot(ax, text_kw, shift, **{**visualdict, **kwargs})


class MetaFlight(type):
    def __getattr__(cls, name):
        if name.startswith("aligned_on_"):
            return lambda flight: cls.aligned_on_ils(flight, name[11:])
        raise AttributeError


class Flight(
    HBoxMixin,
    GeographyMixin,
    ShapelyMixin,
    NavigationFeatures,
    FuzzyLogic,
    metaclass=MetaFlight,
):
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
          `last() <#traffic.core.Flight.last>`_,
          `skip() <#traffic.core.Flight.skip>`_,
          `shorten() <#traffic.core.Flight.shorten>`_
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
        - navigation related method:
          `closest_point() <#traffic.core.Flight.closest_point>`_,
          `takeoff_airport() <#traffic.core.Flight.takeoff_airport>`_,
          `landing_airport() <#traffic.core.Flight.landing_airport>`_,
          `on_runway() <#traffic.core.Flight.on_runway>`_,
          `aligned_on_runway() <#traffic.core.Flight.aligned_on_runway>`_,
          `aligned_on_ils() <#traffic.core.Flight.aligned_on_ils>`_
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

        # This just cannot return None in this case.
        return Traffic.from_flights([self, other])  # type: ignore

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
        if self.diverted is not None:
            title += f"<li><b>diverted to: {self.diverted}</b></li>"
        title += "</ul>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def _repr_svg_(self):
        # even 25m should be enough to limit the size of resulting notebooks!
        if self.shape is None:
            return None

        if len(self.shape.coords) < 1000:
            return super()._repr_svg_()

        return super(
            Flight,
            # cast should be useless but return type of simplify() is Union
            cast(Flight, self.resample("1s").simplify(25)),
        )._repr_svg_()

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

    def __getattr__(self, name: str):
        """Helper to facilitate method chaining without lambda.

        TODO improve the documentation
        flight.altitude_max
            => flight.max('altitude')
        flight.vertical_rate_std
            => flight.std('vertical_rate')

        Flight.feature_gt("altitude_max", 10000)
            => lambda f: f.max('altitude') > 10000
        """
        msg = f"'{self.__class__.__name__}' has no attribute '{name}'"
        if "_" not in name:
            raise AttributeError(msg)
        *name_split, agg = name.split("_")
        feature = "_".join(name_split)
        if feature not in self.data.columns:
            raise AttributeError(msg)
        return getattr(self.data[feature], agg)()

    def filter_if(self, test: Callable[["Flight"], bool]) -> Optional["Flight"]:
        # TODO deprecate if pipe() does a good job?
        return self if test(self) else None

    def has(
        self, method: Union[str, Callable[["Flight"], Iterator["Flight"]]]
    ) -> bool:
        """Returns True if flight.method() returns a non-empty iterator.

        Example usage:

        >>> flight.has("go_around")
        >>> flight.has("runway_change")
        >>> flight.has(lambda f: f.aligned_on_ils("LFBO"))
        """
        return self.next(method) is not None  # noqa: B305

    def sum(
        self, method: Union[str, Callable[["Flight"], Iterator["Flight"]]]
    ) -> int:
        """Returns the number of segments returns by flight.method().

        Example usage:

        >>> flight.sum("go_around")
        >>> flight.sum("runway_change")
        >>> flight.sum(lambda f: f.aligned_on_ils("LFBO"))
        """
        fun = (
            getattr(self.__class__, method)
            if isinstance(method, str)
            else method
        )
        return sum(1 for _ in fun(self))

    def all(
        self, method: Union[str, Callable[["Flight"], Iterator["Flight"]]]
    ) -> Optional["Flight"]:
        """Returns the concatenation of segments returns by flight.method().

        Example usage:

        >>> flight.all("go_around")
        >>> flight.all("runway_change")
        >>> flight.all(lambda f: f.aligned_on_ils("LFBO"))
        """
        fun = (
            getattr(self.__class__, method)
            if isinstance(method, str)
            else method
        )
        t = sum(flight.assign(index_=i) for i, flight in enumerate(fun(self)))
        if t == 0:
            return None
        return Flight(t.data)  # type: ignore

    def next(
        self, method: Union[str, Callable[["Flight"], Iterator["Flight"]]],
    ) -> Optional["Flight"]:
        """
        Returns the first segment of trajectory yielded by flight.method()

        >>> flight.next("go_around")
        >>> flight.next("runway_change")
        >>> flight.next(lambda f: f.aligned_on_ils("LFBO"))
        """
        fun = (
            getattr(self.__class__, method)
            if isinstance(method, str)
            else method
        )
        return next(fun(self), None)

    # --- Iterators ---

    @property
    def timestamp(self) -> Iterator[pd.Timestamp]:
        yield from self.data.timestamp

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        data = self.data.query("longitude == longitude")
        if "altitude" not in data.columns:
            data = data.assign(altitude=0)
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
        if self_filtered is None:
            return None
        iterator = iter(zip(self_filtered.coords, self_filtered.timestamp))
        while True:
            next_ = next(iterator, None)
            if next_ is None:
                return
            coords, time = next_
            yield (coords[0], coords[1], time.to_pydatetime().timestamp())

    # --- Properties (and alike) ---

    def min(self, feature: str):
        """Returns the minimum value of given feature.

        >>> flight.min('altitude')  # dummy example
        24000
        """
        return self.data[feature].min()

    def max(self, feature: str):
        """Returns the maximum value of given feature.

        >>> flight.max('altitude')  # dummy example
        35000
        """
        return self.data[feature].max()

    def mean(self, feature: str):
        """Returns the average value of given feature.

        >>> flight.mean('vertical_rate')  # dummy example
        -1000
        """
        return self.data[feature].mean()

    def feature_gt(
        self,
        feature: Union[str, Callable[["Flight"], Any]],
        value: Any,
        strict: bool = True,
    ) -> bool:
        """Returns True if feature(flight) is greater than value.

        This is fully equivalent to `f.longer_than("1 minute")`:

        >>> f.feature_gt("duration", pd.Timedelta('1 minute'))
        True

        This is equivalent to `f.max('altitude') > 35000`:

        >>> f.feature_gt(lambda f: f.max("altitude"), 35000)
        True

        The second one can be useful for stacking operations during
        lazy evaluation.
        """
        if isinstance(feature, str):
            feature = attrgetter(feature)
        attribute = feature(self)
        if strict:
            return attribute > value
        return attribute >= value

    def feature_lt(
        self,
        feature: Union[str, Callable[["Flight"], Any]],
        value: Any,
        strict: bool = True,
    ) -> bool:
        """Returns True if feature(flight) is less than value.

        This is fully equivalent to `f.shorter_than("1 minute")`:

        >>> f.feature_lt("duration", pd.Timedelta('1 minute'))
        True

        This is equivalent to `f.max('altitude') < 35000`:

        >>> f.feature_lt(lambda f: f.max("altitude"), 35000)
        True

        The second one can be useful for stacking operations during
        lazy evaluation.
        """
        if isinstance(feature, str):
            feature = attrgetter(feature)
        attribute = feature(self)
        if strict:
            return attribute < value
        return attribute <= value

    def shorter_than(
        self, value: Union[str, timedelta, pd.Timedelta], strict: bool = True
    ) -> bool:
        """Returns True if flight duration is shorter than value."""
        if isinstance(value, str):
            value = pd.Timedelta(value)
        return self.feature_lt(attrgetter("duration"), value, strict)

    def longer_than(
        self, value: Union[str, timedelta, pd.Timedelta], strict: bool = True
    ) -> bool:
        """Returns True if flight duration is shorter than value."""
        if isinstance(value, str):
            value = pd.Timedelta(value)
        return self.feature_gt(attrgetter("duration"), value, strict)

    def abs(self, features: Union[str, List[str]], **kwargs) -> "Flight":
        """Assign absolute versions of features to new columns.

        >>> flight.abs("track")

        The two following commands are equivalent:

        >>> flight.abs(["track", "heading"])
        >>> flight.abs(track="track_abs", heading="heading_abs")

        """
        assign_dict = dict()
        if isinstance(features, str):
            features = [features]
        if isinstance(features, Iterable):
            for feature in features:
                assign_dict[feature + "_abs"] = self.data[feature].abs()
        for key, value in kwargs.items():
            assign_dict[value] = self.data[key].abs()
        return self.assign(**assign_dict)

    def diff(self, features: Union[str, List[str]], **kwargs) -> "Flight":
        """Assign differential versions of features to new columns.

        >>> flight.diff("track")

        The two following commands are equivalent:

        >>> flight.diff(["track", "heading"])
        >>> flight.diff(track="track_diff", heading="heading_diff")

        """
        assign_dict = dict()
        if isinstance(features, str):
            features = [features]
        if isinstance(features, Iterable):
            for feature in features:
                assign_dict[feature + "_diff"] = self.data[feature].diff()
        for key, value in kwargs.items():
            assign_dict[value] = self.data[key].diff()
        return self.assign(**assign_dict)

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
            title += f" – {number}"

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
    def diverted(self) -> Union[str, Set[str], None]:
        """Returns the unique diverted value(s),
        None if not available in the DataFrame.

        The diverted airport is usually represented as a ICAO or a IATA code.

        The ICAO code of an airport is represented by 4 letters (e.g. EHAM for
        Amsterdam Schiphol International Airport) and the IATA code is
        represented by 3 letters and more familiar to the public (e.g. AMS for
        Amsterdam)

        """
        return self._get_unique("diverted")

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
        from ..data import aircraft

        if not isinstance(self.icao24, str):
            return None

        res = str(self.icao24)
        ac = aircraft.get_unique(res)

        if ac is None:
            return res

        registration = ac["registration"]
        typecode = ac["typecode"]
        flag = ac["flag"]

        if registration is not None:
            res += f" · {flag} {registration}"
        else:
            res = f"{flag} {res}"

        if typecode is not None:
            res += f" ({typecode})"

        return res

    # -- Time handling, splitting, interpolation and resampling --

    def skip(self, value: deltalike = None, **kwargs) -> Optional["Flight"]:
        """Removes the first n days, hours, minutes or seconds of the Flight.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.skip(minutes=10)
        >>> flight.skip("1H")
        >>> flight.skip(10)  # seconds by default
        """
        delta = to_timedelta(value, **kwargs)
        bound = self.start + delta  # noqa: F841 => used in the query
        # full call is necessary to keep @bound as a local variable
        df = self.data.query("timestamp >= @bound")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def first(self, value: deltalike = None, **kwargs) -> Optional["Flight"]:
        """Returns the first n days, hours, minutes or seconds of the Flight.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.first(minutes=10)
        >>> flight.first("1H")
        >>> flight.first(10)  # seconds by default
        """
        delta = to_timedelta(value, **kwargs)
        bound = self.start + delta  # noqa: F841 => used in the query
        # full call is necessary to keep @bound as a local variable
        df = self.data.query("timestamp < @bound")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def shorten(self, value: deltalike = None, **kwargs) -> Optional["Flight"]:
        """Removes the last n days, hours, minutes or seconds of the Flight.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.shorten(minutes=10)
        >>> flight.shorten("1H")
        >>> flight.shorten(10)  # seconds by default
        """
        delta = to_timedelta(value, **kwargs)
        bound = self.stop - delta  # noqa: F841 => used in the query
        # full call is necessary to keep @bound as a local variable
        df = self.data.query("timestamp <= @bound")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def last(self, value: deltalike = None, **kwargs) -> Optional["Flight"]:
        """Returns the last n days, hours, minutes or seconds of the Flight.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        >>> flight.last(minutes=10)
        >>> flight.last("1H")
        >>> flight.last(10)  # seconds by default
        """
        delta = to_timedelta(value, **kwargs)
        bound = self.stop - delta  # noqa: F841 => used in the query
        # full call is necessary to keep @bound as a local variable
        df = self.data.query("timestamp > @bound")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def before(self, time: timelike, strict: bool = True) -> Optional["Flight"]:
        """Returns the part of the trajectory flown before a given timestamp.

        - ``time`` can be passed as a string, an epoch, a Python datetime, or
          a Pandas timestamp.
        """
        return self.between(self.start, time, strict)

    def after(self, time: timelike, strict: bool = True) -> Optional["Flight"]:
        """Returns the part of the trajectory flown after a given timestamp.

        - ``time`` can be passed as a string, an epoch, a Python datetime, or
          a Pandas timestamp.
        """
        return self.between(time, self.stop, strict)

    def between(
        self, start: timelike, stop: time_or_delta, strict: bool = True
    ) -> Optional["Flight"]:
        """Returns the part of the trajectory flown between start and stop.

        - ``start`` and ``stop`` can be passed as a string, an epoch, a Python
          datetime, or a Pandas timestamp.
        - ``stop`` can also be passed as a timedelta.

        """

        # Corner cases when start or stop are None or NaT
        if start is None or start != start:
            return self.before(stop, strict=strict)

        if stop is None or stop != stop:
            return self.after(start, strict=strict)

        start = to_datetime(start)
        if isinstance(stop, timedelta):
            stop = start + stop
        else:
            stop = to_datetime(stop)

        # full call is necessary to keep @start and @stop as local variables
        # return self.query('@start < timestamp < @stop')  => not valid
        if strict:
            df = self.data.query("@start < timestamp < @stop")
        else:
            df = self.data.query("@start <= timestamp <= @stop")

        if df.shape[0] == 0:
            return None

        return self.__class__(df)

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
        if ratio < 0 or ratio > 1:
            raise RuntimeError("ratio must be comprised between 0 and 1")

        subset = self.between(
            self.start, self.start + ratio * self.duration, strict=False
        )

        assert subset is not None
        return subset.at()

    @flight_iterator
    def sliding_windows(
        self, duration: deltalike, step: deltalike,
    ) -> Iterator["Flight"]:

        duration_ = to_timedelta(duration)
        step_ = to_timedelta(step)

        first = self.first(duration_)
        if first is None:
            return

        yield first

        after = self.after(self.start + step_)
        if after is not None:
            yield from after.sliding_windows(duration_, step_)

    @overload
    def split(self, value: int, unit: str) -> FlightIterator:
        ...

    @overload
    def split(  # noqa: F811
        self, value: str, unit: None = None
    ) -> FlightIterator:
        ...

    @flight_iterator
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
        if isinstance(value, int) and unit is None:
            # default value is 10 m
            unit = "m"

        for data in _split(self.data, value, unit):
            yield self.__class__(data)

    def max_split(
        self,
        value: Union[int, str] = "10T",
        unit: Optional[str] = None,
        key: Callable[[Optional["Flight"]], Any] = attrgetter_duration,
    ) -> Optional["Flight"]:
        """Returns the biggest (by default, longest) part of trajectory.

        Example usage:

        >>> from traffic.data.samples import elal747
        >>> elal747.query("altitude < 15000").max_split()
        Flight ELY1747
        aircraft: 738043 · 🇮🇱 4X-ELC (B744)
        origin: LIRF (2019-11-03 12:14:40+00:00)
        destination: LLBG (2019-11-03 14:13:00+00:00)

        In this example, the fancy part of the trajectory occurs below
        15,000 ft. The command extracts the plane pattern.

        """

        # warnings.warn("Use split().max() instead.", DeprecationWarning)
        return max(
            self.split(value, unit),  # type: ignore
            key=key,
            default=None,
        )

    def apply_segments(
        self, fun: Callable[..., "LazyTraffic"], name: str, *args, **kwargs
    ) -> Optional["Flight"]:
        return getattr(self, name)(*args, **kwargs)(fun)

    def apply_time(
        self, freq: str = "1T", merge: bool = True, **kwargs,
    ) -> "Flight":
        """Apply features on time windows.

        The following is performed:

        - a new column `rounded` rounds the timestamp at the given rate;
        - the groupby/apply is operated with parameters passed in apply;
        - if merge is True, the new column in merged into the Flight,
          otherwise a pd.DataFrame is returned.

        For example:

        >>> f.agg_time("10T", straight=lambda df: Flight(df).distance())

        returns a Flight with a new column straight with the great circle
        distance between points sampled every 10 minutes.
        """

        if len(kwargs) == 0:
            raise RuntimeError("No feature provided for aggregation.")
        temp_flight = self.assign(
            rounded=lambda df: df.timestamp.dt.round(freq)
        )

        agg_data = None

        for label, fun in kwargs.items():
            agg_data = (
                agg_data.merge(  # type: ignore
                    temp_flight.groupby("rounded")
                    .apply(lambda df: fun(self.__class__(df)))
                    .rename(label),
                    left_index=True,
                    right_index=True,
                )
                if agg_data is not None
                else temp_flight.groupby("rounded")
                .apply(lambda df: fun(self.__class__(df)))
                .rename(label)
                .to_frame()
            )

        if not merge:  # mostly for debugging purposes
            return agg_data  # type: ignore

        return temp_flight.merge(agg_data, left_on="rounded", right_index=True)

    def agg_time(
        self, freq: str = "1T", merge: bool = True, **kwargs,
    ) -> "Flight":
        """Aggregate features on time windows.

        The following is performed:

        - a new column `rounded` rounds the timestamp at the given rate;
        - the groupby/agg is operated with parameters passed in kwargs;
        - if merge is True, the new column in merged into the Flight,
          otherwise a pd.DataFrame is returned.

        For example:

        >>> f.agg_time('3T', groundspeed='mean')

        returns a Flight with a new column groundspeed_mean with groundspeed
        averaged per intervals of 3 minutes.
        """

        def flatten(
            data: pd.DataFrame, how: Callable = "_".join
        ) -> pd.DataFrame:
            data.columns = (
                [
                    how(filter(None, map(str, levels)))
                    for levels in data.columns.values
                ]
                if isinstance(data.columns, pd.MultiIndex)
                else data.columns
            )
            return data

        if len(kwargs) == 0:
            raise RuntimeError("No feature provided for aggregation.")
        temp_flight = self.assign(
            rounded=lambda df: df.timestamp.dt.round(freq)
        )

        # force the agg_data to be multi-indexed in columns
        kwargs_modified: Dict["str", List[Any]] = dict(
            (
                key,
                list(value)
                if any(isinstance(value, x) for x in [list, tuple])
                else [value],
            )
            for key, value in kwargs.items()
        )
        agg_data = flatten(temp_flight.groupby("rounded").agg(kwargs_modified))

        if not merge:  # mostly for debugging purposes
            return agg_data

        return temp_flight.merge(agg_data, left_on="rounded", right_index=True)

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
                .unwrap()  # avoid filled gaps in track and heading
                .assign(start=self.start, stop=self.stop)
                .data.set_index("timestamp")
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
                .unwrap()  # avoid filled gaps in track and heading
                .assign(tz_naive=lambda d: d.timestamp.astype("datetime64[ns]"))
                .data.set_index("tz_naive")
                .asfreq((self.stop - self.start) / (rule - 1), method="nearest")
                .reset_index()
                .drop(columns="tz_naive")
            )
        else:
            raise TypeError("rule must be a str or an int")

        if "track_unwrapped" in data.columns:
            data = data.assign(track=lambda df: df.track_unwrapped % 360)
        if "heading_unwrapped" in data.columns:
            data = data.assign(heading=lambda df: df.heading_unwrapped % 360)

        return self.__class__(data)

    def filter(
        self,
        strategy: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = lambda x: x.bfill().ffill(),
        **kwargs,
    ) -> "Flight":

        """Filters the trajectory given features with a median filter.

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

        ks_dict: Dict[str, Union[int, Iterable[int]]] = {
            "altitude": (17, 53),
            "selected_mcp": (17, 53),
            "selected_fms": (17, 53),
            "IAS": 23,
            "TAS": 23,
            "Mach": 23,
            "groundspeed": 5,
            "compute_gs": (17, 53),
            "compute_track": 17,
            "onground": 3,
            "latitude": 1,  # the method doesn't apply well to positions
            "longitude": 1,
            **kwargs,
        }

        if strategy is None:

            def identity(x):
                return x  # noqa: E704

            strategy = identity

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

        kernels_size: List[Union[int, Iterable[int]]] = [0 for _ in features]
        for idx, feature in enumerate(features):
            kernels_size[idx] = ks_dict.get(feature, 17)

        for feat, ks_list in zip(features, kernels_size):

            if isinstance(ks_list, int):
                ks_list = [ks_list]
            else:
                ks_list = list(ks_list)

            for ks in ks_list:
                # Prepare each feature for the filtering
                df = cascaded_filters(new_data[["timestamp", feat]], feat, ks)

                # Decision to accept/reject data points in the time series
                new_data.loc[df.sq_eps > df.sigma, feat] = None

        data = strategy(new_data)

        if "onground" in data.columns:
            data = data.assign(onground=data.onground.astype(bool))

        return self.__class__(data)

    def filter_position(self, cascades: int = 2) -> Optional["Flight"]:
        # TODO improve based on agg_time
        flight: Optional["Flight"] = self
        for _ in range(cascades):
            if flight is None:
                return None
            flight = flight.cumulative_distance().query(
                "compute_gs < compute_gs.mean() + 3 * compute_gs.std()"
            )
        return flight

    def comet(self, **kwargs) -> "Flight":
        """Computes a comet for a trajectory.

        The method uses the last position of a trajectory (method `at()
        <#traffic.core.Flight.at>`_) and uses the ``track`` (in degrees),
        ``groundspeed`` (in knots) and ``vertical_rate`` (in ft/min) values to
        interpolate the trajectory in a straight line.

        The elements passed as kwargs as passed as is to the datetime.timedelta
        constructor.

        Example usage:

        .. code:: python

            flight.comet(minutes=10)
            flight.before("2018-12-24 23:55").comet(minutes=10)  # Merry XMas!

        """

        last_line = self.at()
        if last_line is None:
            raise ValueError("Unknown data for this flight")
        window = self.last(seconds=20)
        delta = timedelta(**kwargs)

        if window is None:
            raise RuntimeError("Flight expect at least 20 seconds of data")

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

    # TODO change to Iterator
    def onground(self) -> Optional["Flight"]:
        if "altitude" not in self.data.columns:
            return self
        if "onground" in self.data.columns and self.data.onground.dtype == bool:
            return self.query("onground or altitude != altitude")
        else:
            return self.query("altitude != altitude")

    def airborne(self) -> Optional["Flight"]:
        """Returns the airborne part of the Flight.

        The airborne part is determined by an ``onground`` flag or null values
        in the altitude column.
        """
        if "altitude" not in self.data.columns:
            return None
        if "onground" in self.data.columns and self.data.onground.dtype == bool:
            return self.query("not onground and altitude == altitude")
        else:
            return self.query("altitude == altitude")

    def unwrap(self, features: Union[None, str, List[str]] = None) -> "Flight":
        """Unwraps angles in the DataFrame.

        All features representing angles may be unwrapped (through Numpy) to
        avoid gaps between 359° and 1°.

        The method applies by default to features ``track`` and ``heading``.
        More or different features may be passed in parameter.
        """
        if features is None:
            features = default_angle_features

        if isinstance(features, str):
            features = [features]

        reset = self.reset_index(drop=True)

        result_dict = dict()
        for feature in features:
            if feature not in reset.data.columns:
                continue
            series = reset.data[feature]
            idx = ~series.isnull()
            result_dict[f"{feature}_unwrapped"] = pd.Series(
                np.degrees(np.unwrap(np.radians(series.loc[idx]))),
                index=series.loc[idx].index,
            )

        return reset.assign(**result_dict)

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

        if any(w not in self.data.columns for w in ["heading", "TAS"]):
            raise RuntimeError(
                "No wind data in trajectory. Consider Flight.query_ehs()"
            )

        return self.assign(
            wind_u=self.data.groundspeed * np.sin(np.radians(self.data.track))
            - self.data.TAS * np.sin(np.radians(self.data.heading)),
            wind_v=self.data.groundspeed * np.cos(np.radians(self.data.track))
            - self.data.TAS * np.cos(np.radians(self.data.heading)),
        )

    def plot_wind(
        self,
        ax: GeoAxesSubplot,
        resolution: Union[int, str, Dict[str, float], None] = "5T",
        filtered: bool = False,
        **kwargs,
    ) -> List[Artist]:  # coverage: ignore
        """Plots the wind field seen by the aircraft on a Matplotlib axis.

        The Flight supports Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        The `resolution` argument may be:

            - None for a raw plot;
            - an integer or a string to pass to a `Flight.resample()
              <#traffic.core.Flight.resample>`__ method as a preprocessing
              before plotting;
            - or a dictionary, e.g dict(latitude=4, longitude=4), if you
              want a grid with a resolution of 4 points per latitude and
              longitude degree.

        Example usage:

        .. code:: python

            from traffic.drawing import Mercator
            fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator()))
            (
                flight
                .resample("1s")
                .query('altitude > 10000')
                .compute_wind()
                .plot_wind(ax, alpha=.5)
            )

        """

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            kwargs["transform"] = PlateCarree()

        if any(w not in self.data.columns for w in ["wind_u", "wind_v"]):
            raise RuntimeError(
                "No wind data in trajectory. Consider Flight.compute_wind()"
            )

        copy_self: Optional[Flight] = self

        if filtered:
            copy_self = self.filter(roll=17)
            if copy_self is None:
                return []
            copy_self = copy_self.query("roll.abs() < .5")
            if copy_self is None:
                return []
            copy_self = copy_self.filter(wind_u=17, wind_v=17)

        if copy_self is None:
            return []

        if resolution is not None:

            if isinstance(resolution, (int, str)):
                data = copy_self.resample(resolution).data

            if isinstance(resolution, dict):
                r_lat = resolution.get("latitude", None)
                r_lon = resolution.get("longitude", None)

                if r_lat is not None and r_lon is not None:
                    data = (
                        copy_self.assign(
                            latitude=lambda x: (
                                (r_lat * x.latitude).round() / r_lat
                            ),
                            longitude=lambda x: (
                                (r_lon * x.longitude).round() / r_lon
                            ),
                        )
                        .groupby(["latitude", "longitude"])
                        .agg(dict(wind_u="mean", wind_v="mean"))
                        .reset_index()
                    )

        return ax.barbs(
            data.longitude.values,
            data.latitude.values,
            data.wind_u.values,
            data.wind_v.values,
            **kwargs,
        )

    # -- Distances --

    def bearing(
        self, other: PointMixin, column_name: str = "bearing"
    ) -> "Flight":
        # temporary, should implement full stuff
        size = len(self)
        return self.assign(
            **{
                column_name: geo.bearing(
                    self.data.latitude.values,
                    self.data.longitude.values,
                    other.latitude * np.ones(size),
                    other.longitude * np.ones(size),
                )
                % 360
            }
        )

    @overload
    def distance(  # type: ignore
        self, other: None = None, column_name: str = "distance"
    ) -> float:
        ...

    @overload
    def distance(  # noqa: F811
        self,
        other: Union["Airspace", Polygon, PointMixin],
        column_name: str = "distance",
    ) -> "Flight":
        ...

    @overload
    def distance(  # noqa: F811
        self, other: "Flight", column_name: str = "distance"
    ) -> Optional[pd.DataFrame]:
        ...

    def distance(  # noqa: F811
        self,
        other: Union[None, "Flight", "Airspace", Polygon, PointMixin] = None,
        column_name: str = "distance",
    ) -> Union[None, float, "Flight", pd.DataFrame]:

        """Computes the distance from a Flight to another entity.

        The behaviour is different according to the type of the second
        element:

        - if the other element is None (i.e. flight.distance()), the method
          returns a distance in nautical miles between the first and last
          recorded positions in the DataFrame.

        - if the other element is a Flight, the method returns a pandas
          DataFrame with corresponding data from both flights, aligned
          with their timestamps, and two new columns with `lateral` and
          `vertical` distances (resp. in nm and ft) separating them.

        - otherwise, the same Flight is returned enriched with a new
          column (by default, named "distance") with the distance of each
          point of the trajectory to the geometrical element.

        .. warning::

            - An Airspace is (currently) considered as its flattened
              representation
            - Computing a distance to a polygon is quite slow at the moment.
              Consider a strict resampling (e.g. one point per minute, "1T")
              before calling the method.

        """

        if other is None:
            first = self.at_ratio(0)
            last = self.at_ratio(1)
            if first is None or last is None:
                return 0
            return (
                geo.distance(
                    first.latitude,
                    first.longitude,
                    last.latitude,
                    last.longitude,
                )
                / 1852  # in nautical miles
            )

        if isinstance(other, PointMixin):
            size = len(self)
            return self.assign(
                **{
                    column_name: geo.distance(
                        self.data.latitude.values,
                        self.data.longitude.values,
                        other.latitude * np.ones(size),
                        other.longitude * np.ones(size),
                    )
                    / 1852  # in nautical miles
                }
            )

        from .airspace import Airspace  # noqa: F811

        if isinstance(other, Airspace):
            other = other.flatten()

        if isinstance(other, Polygon):
            bounds = other.bounds

            projection = pyproj.Proj(
                proj="aea",  # equivalent projection
                lat_1=bounds[1],
                lat_2=bounds[3],
                lat_0=(bounds[1] + bounds[3]) / 2,
                lon_0=(bounds[0] + bounds[2]) / 2,
            )

            transformer = pyproj.Transformer.from_proj(
                pyproj.Proj("epsg:4326"), projection, always_xy=True
            )
            projected_shape = transform(transformer.transform, other)

            self_xy = self.compute_xy(projection)

            return self.assign(
                **{
                    column_name: list(
                        projected_shape.exterior.distance(p)
                        * (-1 if projected_shape.contains(p) else 1)
                        for p in MultiPoint(
                            list(zip(self_xy.data.x, self_xy.data.y))
                        )
                    )
                }
            )

        start = max(self.start, other.start)
        stop = min(self.stop, other.stop)
        f1, f2 = (self.between(start, stop), other.between(start, stop))
        if f1 is None or f2 is None:
            return None

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
        self,
        compute_gs: bool = True,
        compute_track: bool = True,
        *,
        reverse: bool = False,
        **kwargs,
    ) -> "Flight":

        """ Enrich the structure with new ``cumdist`` column computed from
        latitude and longitude columns.

        The first ``cumdist`` value is 0, then distances are computed (in
        **nautical miles**) and summed between consecutive positions. The last
        value is the total length of the trajectory.

        When the ``compute_gs`` flag is set to True (default), an additional
        ``compute_gs`` is also added. This value can be compared with the
        decoded ``groundspeed`` value in ADSB messages.

        When the ``compute_track`` flag is set to True (default), an additional
        ``compute_track`` is also added. This value can be compared with the
        decoded ``track`` value in ADSB messages.

        """

        if "compute_groundspeed" in kwargs:
            warnings.warn("Use compute_gs argument", DeprecationWarning)
            compute_gs = kwargs["compute_groundspeed"]

        cur_sorted = self.sort_values("timestamp", ascending=not reverse)
        coords = cur_sorted.data[["timestamp", "latitude", "longitude"]]

        delta = pd.concat([coords, coords.add_suffix("_1").diff()], axis=1)
        delta_1 = delta.iloc[1:]
        d = geo.distance(
            delta_1.latitude.values,
            delta_1.longitude.values,
            (delta_1.latitude + delta_1.latitude_1).values,
            (delta_1.longitude + delta_1.longitude_1).values,
        )

        res = cur_sorted.assign(
            cumdist=np.pad(d.cumsum() / 1852, (1, 0), "constant")
        )

        if compute_gs:
            gs = d / delta_1.timestamp_1.dt.total_seconds() * (3600 / 1852)
            res = res.assign(compute_gs=np.abs(np.pad(gs, (1, 0), "edge")))

        if compute_track:
            track = geo.bearing(
                delta_1.latitude.values,
                delta_1.longitude.values,
                (delta_1.latitude + delta_1.latitude_1).values,
                (delta_1.longitude + delta_1.longitude_1).values,
            )
            track = np.where(track > 0, track, 360 + track)
            res = res.assign(
                compute_track=np.abs(np.pad(track, (1, 0), "edge"))
            )

        return res.sort_values("timestamp", ascending=True)

    # -- Geometry operations --

    @property
    def linestring(self) -> Optional[LineString]:
        # longitude is implicit I guess
        if "latitude" not in self.data.columns:
            return None
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

    def intersects(self, shape: Union[ShapelyMixin, base.BaseGeometry]) -> bool:
        # implemented and monkey-patched in airspace.py
        # given here for consistency in types
        ...

    def clip(
        self, shape: Union[ShapelyMixin, base.BaseGeometry]
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
        list_coords = list(self.xy_time)
        if len(list_coords) < 2:
            return None

        linestring = LineString(list_coords)
        if not isinstance(shape, base.BaseGeometry):
            shape = shape.shape

        intersection = linestring.intersection(shape)

        if intersection.is_empty:
            return None

        if isinstance(intersection, Point):
            return None

        if isinstance(intersection, LineString):
            time_list = list(
                datetime.fromtimestamp(t, timezone.utc)
                for t in np.stack(intersection.coords)[:, 2]
            )
            return self.between(min(time_list), max(time_list))

        def _clip_generator() -> Iterable[Tuple[datetime, datetime]]:
            for segment in intersection:
                times: List[datetime] = list(
                    datetime.fromtimestamp(t, timezone.utc)
                    for t in np.stack(segment.coords)[:, 2]
                )
                yield min(times), max(times)

        times: List[Tuple[datetime, datetime]] = list(_clip_generator())
        clipped_flight = self.between(
            min(t for t, _ in times), max(t for _, t in times)
        )

        if clipped_flight is None:
            return None

        if clipped_flight.shape is None:
            return None

        return clipped_flight

    # -- OpenSky specific methods --

    def query_opensky_sensors(self, where_condition: str = "") -> pd.DataFrame:
        from ..data import opensky

        return (
            opensky.request(
                "select s.ITEM, count(*) from state_vectors_data4, "
                "state_vectors_data4.serials s "
                f"where icao24='{self.icao24}' and "
                f"{where_condition} "
                "{before_time}<=time and time < {after_time} and "
                "{before_hour}<=hour and hour < {after_hour} "
                "group by s.ITEM;",
                self.start,
                self.stop,
                columns=["serial", "count"],
            )
            .groupby("serial")
            .sum()
        )

    def query_opensky(self, **kwargs) -> Optional["Flight"]:
        """Returns data from the same Flight as stored in OpenSky database.

        This may be useful if you write your own parser for data from a
        different channel. The method will use the ``callsign`` and ``icao24``
        attributes to build a request for current Flight in the OpenSky Network
        database.

        The kwargs argument helps overriding arguments from the query, namely
        start, stop, callsign and icao24.

        Returns None if no data is found.

        .. note::
            Read more about access to the OpenSky Network database `here
            <opensky_impala.html>`_
        """

        from ..data import opensky

        query_params = {
            "start": self.start,
            "stop": self.stop,
            "callsign": self.callsign,
            "icao24": self.icao24,
            "return_flight": True,
            **kwargs,
        }
        return opensky.history(**query_params)  # type: ignore

    def query_ehs(
        self,
        data: Union[None, pd.DataFrame, "RawData"] = None,
        failure_mode: str = "warning",
        progressbar: Union[bool, Callable[[Iterable], Iterable]] = True,
    ) -> "Flight":
        """Extends data with extra columns from EHS messages.

        By default, raw messages are requested from the OpenSky Network
        database.

        .. warning::
            Making a lot of small requests can be very inefficient and may look
            like a denial of service. If you get the raw messages using a
            different channel, you can provide the resulting dataframe as a
            parameter. See the page about `OpenSky Impala access
            <opensky_impala.html>`_

        The data parameter expect three columns: ``icao24``, ``rawmsg`` and
        ``mintime``, in conformance with the OpenSky API.

        .. note::
            Read more about access to the OpenSky Network database `here
            <opensky_impala.html>`_
        """

        from ..data import opensky
        from ..data.adsb.raw_data import RawData  # noqa: F811

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
            ext = opensky.extended(self.start, self.stop, icao24=self.icao24)
            df = ext.data if ext is not None else None
        else:
            df = data if isinstance(data, pd.DataFrame) else data.data
            df = df.query(
                "icao24 == @self.icao24 and "
                "@self.start < mintime < @self.stop"
            )

        if df is None or df.shape[0] == 0:
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

        t = RawData(referenced_df).decode(
            reference=self.origin if isinstance(self.origin, str) else None,
            progressbar=progressbar,
            progressbar_kw=dict(leave=False, desc=f"{identifier}:"),
        )

        extended = t[self.icao24] if t is not None else None
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

        .. code:: python

            from traffic.drawing import Mercator
            fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator())
            flight.plot(ax, alpha=.5)

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
        self,
        y: Union[str, List[str], alt.Y],
        x: Union[str, alt.X] = "timestamp:T",
        **kwargs,
    ) -> alt.Chart:  # coverage: ignore
        """Plots the given features according to time.

        The method ensures:

        - only non-NaN data are displayed (no gap in the plot);
        - the timestamp is naively converted to UTC if not localized.

        Example usage:

        .. code:: python

            flight.encode('altitude')
            # or with several comparable features
            flight.encode(['groundspeed', 'IAS', 'TAS'])

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
                x=x,
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

        .. code:: python

            ax = plt.axes()
            # most simple version
            flight.plot(ax, 'altitude')
            # or with several comparable features and twin axes
            flight.plot(
                ax, ['altitude', 'groundspeed, 'IAS', 'TAS'],
                secondary_y=['altitude']
            )

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
