from __future__ import annotations

import ast
import logging
import warnings
from datetime import datetime, timedelta, timezone
from functools import lru_cache, reduce
from itertools import combinations
from operator import attrgetter
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import rich.repr
from ipyleaflet import Map as LeafletMap
from ipyleaflet import Polyline as LeafletPolyline
from ipywidgets import HTML
from rich.console import Console, ConsoleOptions, RenderResult

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
from pandas.core.internals import DatetimeTZBlock
from shapely.geometry import LineString, MultiPoint, Point, Polygon, base
from shapely.ops import transform

from ..algorithms.douglas_peucker import douglas_peucker
from ..algorithms.navigation import NavigationFeatures
from ..algorithms.openap import OpenAP
from ..core.structure import Airport
from ..core.types import ProgressbarType
from . import geodesy as geo
from .iterator import FlightIterator, flight_iterator
from .mixins import GeographyMixin, HBoxMixin, PointMixin, ShapelyMixin
from .time import deltalike, time_or_delta, timelike, to_datetime, to_timedelta

if TYPE_CHECKING:
    import altair as alt  # noqa: F401
    from cartopy import crs  # noqa: F401
    from cartopy.mpl.geoaxes import GeoAxesSubplot  # noqa: F401
    from matplotlib.artist import Artist  # noqa: F401
    from matplotlib.axes._subplots import Axes  # noqa: F401

    from ..data.adsb.raw_data import RawData  # noqa: F401
    from ..data.basic.aircraft import Tail  # noqa: F401
    from ..data.basic.navaid import Navaids  # noqa: F401
    from .airspace import Airspace  # noqa: F401
    from .lazy import LazyTraffic  # noqa: F401
    from .structure import Navaid  # noqa: F401
    from .traffic import Traffic  # noqa: F401

_log = logging.getLogger(__name__)


class Entry(TypedDict, total=False):
    timestamp: pd.Timestamp
    timedelta: pd.Timedelta
    longitude: float
    latitude: float
    altitude: float
    name: str


T = TypeVar("T", bound="Flight")

if str(pd.__version__) < "1.3":

    def _tz_interpolate(
        data: DatetimeTZBlock, *args: Any, **kwargs: Any
    ) -> DatetimeTZBlock:
        return data.astype(int).interpolate(*args, **kwargs).astype(data.dtype)

    DatetimeTZBlock.interpolate = _tz_interpolate

else:
    # - with version 1.3.0, interpolate returns a list
    # - Windows require "int64" as "int" may be interpreted as "int32" and raise
    #   an error (was not raised before 1.3.0)

    def _tz_interpolate(
        data: DatetimeTZBlock, *args: Any, **kwargs: Any
    ) -> DatetimeTZBlock:
        coerced = data.coerce_to_target_dtype("int64")
        interpolated, *_ = coerced.interpolate(*args, **kwargs)
        return interpolated

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


class Position(PointMixin, pd.core.series.Series):  # type: ignore
    def plot(
        self,
        ax: "Axes",
        text_kw: Optional[Dict[str, Any]] = None,
        shift: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List["Artist"]:  # coverage: ignore

        from ..drawing.markers import aircraft as aircraft_marker
        from ..drawing.markers import rotate_marker

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
    def __getattr__(cls, name: str) -> Callable[..., Any]:

        # if the string is callable, apply this on a flight
        # parsing the AST is a much safer option than raw eval()
        for node in ast.walk(ast.parse(name)):
            if isinstance(node, ast.Call):
                func_name = node.func.id  # type: ignore
                args = [ast.literal_eval(arg) for arg in node.args]
                kwargs = dict(
                    (keyword.arg, ast.literal_eval(keyword.value))
                    for keyword in node.keywords
                )
                return lambda flight: getattr(Flight, func_name)(
                    flight, *args, **kwargs
                )

        # We should think about deprecating what comes below...
        if name.startswith("aligned_on_"):
            return lambda flight: cls.aligned_on_ils(flight, name[11:])
        if name.startswith("takeoff_runway_"):
            return lambda flight: cls.takeoff_from_runway(flight, name[15:])
        if name.startswith("on_parking_"):
            return lambda flight: cls.on_parking_position(flight, name[11:])
        if name.startswith("pushback_"):
            return lambda flight: cls.pushback(flight, name[9:])
        if name.startswith("landing_at_"):
            return lambda flight: cls.landing_at(flight, name[11:])
        if name.startswith("takeoff_from_"):
            return lambda flight: cls.takeoff_from(flight, name[13:])

        raise AttributeError


@rich.repr.auto()
class Flight(
    HBoxMixin,
    GeographyMixin,
    ShapelyMixin,
    NavigationFeatures,
    OpenAP,
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

    .. note::

        All navigation related methods are described more in depth on a
        `dedicated page <navigation.html>`_.

    **Abridged contents:**

        - properties:
          :meth:`callsign`,
          :meth:`flight_id`,
          :meth:`icao24`,
          :meth:`number`,
          :meth:`start`,
          :meth:`stop`,

        - time related methods:
          :meth:`after`,
          :meth:`at`,
          :meth:`at_ratio`,
          :meth:`before`,
          :meth:`between`,
          :meth:`first`,
          :meth:`last`,
          :meth:`skip`,
          :meth:`shorten`

        - geometry related methods:
          :meth:`airborne`,
          :meth:`clip`,
          :meth:`compute_wind`,
          :meth:`compute_xy`,
          :meth:`distance`,
          :meth:`inside_bbox`,
          :meth:`intersects`,
          :meth:`project_shape`,
          :meth:`simplify`,
          :meth:`unwrap`

        - filtering and resampling methods:
          :meth:`comet`,
          :meth:`filter`,
          :meth:`resample`

        - TMA events:
          :meth:`takeoff_from_runway`,
          :meth:`aligned_on_ils`,
          :meth:`go_around`,
          :meth:`runway_change`

        - airborne events:
          :meth:`aligned_on_navpoint`,
          :meth:`compute_navpoints`,
          :meth:`emergency`

        - ground trajectory methods:
          :meth:`aligned_on_runway`,
          :meth:`on_parking_position`,
          :meth:`pushback`,
          :meth:`slow_taxi`,
          :meth:`moving`

        - visualisation with altair:
          :meth:`chart`,
          :meth:`geoencode`

        - visualisation with leaflet: :meth:`map_leaflet`
        - visualisation with Matplotlib:
          :meth:`plot`,
          :meth:`plot_time`

    .. tip::

        :ref:`Sample flights <How to access sample trajectories?>` are provided
        for testing purposes in module ``traffic.data.samples``

    """

    __slots__ = ("data",)

    # --- Special methods ---

    def __add__(
        self, other: Union[Literal[0], "Flight", "Traffic"]
    ) -> "Traffic":
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

    def __radd__(
        self, other: Union[Literal[0], "Flight", "Traffic"]
    ) -> "Traffic":
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
            data = self.data.drop_duplicates("last_position")
            return data.shape[0]  # type: ignore
        else:
            return self.data.shape[0]  # type: ignore

    @property
    def count(self) -> int:
        return len(self)

    def _info_html(self) -> str:
        title = "<h4><b>Flight</b>"
        if self.flight_id:
            title += f" {self.flight_id}"
        title += "</h4>"

        aircraft_fmt = "<code>%icao24</code> · %flag %registration (%typecode)"

        title += "<ul>"
        title += f"<li><b>callsign:</b> {self.callsign} {self.trip}</li>"
        if self.aircraft is not None:
            title += "<li><b>aircraft:</b> {aircraft}</li>".format(
                aircraft=format(self.aircraft, aircraft_fmt)
            )
        else:
            title += f"<li><b>aircraft:</b> <code>{self.icao24}</code></li>"
        title += f"<li><b>start:</b> {self.start}</li>"
        title += f"<li><b>stop:</b> {self.stop}</li>"
        title += f"<li><b>duration:</b> {self.duration}</li>"

        if self.diverted is not None:
            title += f"<li><b>diverted to: {self.diverted}</b></li>"

        sampling_rate = self.data.timestamp.diff().mean().total_seconds()
        title += f"<li><b>sampling rate:</b> {sampling_rate:.0f} second(s)</li>"

        title += "</ul>"
        return title

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        aircraft_fmt = "%icao24 · %flag %registration (%typecode)"

        yield f"[bold blue]Flight {self.flight_id if self.flight_id else ''}"

        yield f"  - [b]callsign:[/b] {self.callsign} {self.trip}"
        if self.aircraft is not None:
            yield "  - [b]aircraft:[/b] {aircraft}".format(
                aircraft=format(self.aircraft, aircraft_fmt)
            )
        else:
            yield f"  - [b]aircraft:[/b] {self.icao24}"

        yield f"  - [b]start:[/b] {self.start:%Y-%m-%d %H:%M:%S}Z "
        yield f"  - [b]stop:[/b] {self.stop:%Y-%m-%d %H:%M:%S}Z"
        yield f"  - [b]duration:[/b] {self.duration}"

        sampling_rate = self.data.timestamp.diff().mean().total_seconds()
        yield f"  - [b]sampling rate:[/b] {sampling_rate:.0f} second(s)"

        features = set(self.data.columns) - {
            "start",
            "stop",
            "icao24",
            "callsign",
            "flight_id",
            "destination",
            "origin",
            "track_unwrapped",
            "heading_unwrapped",
        }
        yield "  - [b]features:[/b]"
        for feat in sorted(features):
            yield f"    o {feat}, [i]{self.data[feat].dtype}"

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    @lru_cache()
    def _repr_svg_(self) -> Optional[str]:
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

    def __rich_repr__(self) -> rich.repr.Result:
        if self.flight_id:
            yield self.flight_id
        yield "icao24", self.icao24
        yield "callsign", self.callsign

    @property
    def __geo_interface__(self) -> Dict[str, Any]:
        if self.shape is None:
            # Returns an empty geometry
            return {"type": "GeometryCollection", "geometries": []}
        return self.shape.__geo_interface__  # type: ignore

    def keys(self) -> list[str]:
        # This is for allowing dict(Flight)
        keys = ["callsign", "icao24", "aircraft", "start", "stop", "duration"]
        if self.flight_id:
            keys = ["flight_id"] + keys
        if self.origin:
            keys.append("origin")
        if self.destination:
            keys.append("destination")
        if self.diverted:
            keys.append("diverted")
        return keys

    def __getitem__(self, name: str) -> Any:
        if name in self.keys():
            return getattr(self, name)
        raise NotImplementedError()

    def __getattr__(self, name: str) -> Any:
        """Helper to facilitate method chaining without lambda.

        Example usage:

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

    def pipe(
        self,
        func: Callable[..., None | "Flight" | bool],
        *args: Any,
        **kwargs: Any,
    ) -> None | "Flight" | bool:
        """
        Applies `func` to the object.

        .. warning::

            The logic is similar to that of :meth:`~pandas.DataFrame.pipe`
            method, but the function applies on T, not on the DataFrame.

        """
        return func(self, *args, **kwargs)

    def filter_if(self, test: Callable[["Flight"], bool]) -> Optional["Flight"]:
        _log.warning("Use Flight.pipe(...) instead", DeprecationWarning)
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
        """Returns the number of segments returned by flight.method().

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
        self,
        method: Union[str, Callable[["Flight"], Iterator["Flight"]]],
        flight_id: None | str = None,
    ) -> Optional["Flight"]:
        """Returns the concatenation of segments returned by flight.method().

        Example usage:

        >>> flight.all("go_around")
        >>> flight.all("runway_change")
        >>> flight.all('aligned_on_ils("LFBO")')
        >>> flight.all(lambda f: f.aligned_on_ils("LFBO"))
        """
        fun = (
            getattr(self.__class__, method)
            if isinstance(method, str)
            else method
        )
        if flight_id is None:
            t = sum(
                flight.assign(index_=i) for i, flight in enumerate(fun(self))
            )
        else:
            t = sum(
                flight.assign(flight_id=flight_id.format(self=flight, i=i))
                for i, flight in enumerate(fun(self))
            )
        if t == 0:
            return None
        return Flight(t.data)

    def next(
        self,
        method: Union[str, Callable[["Flight"], Iterator["Flight"]]],
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

    def final(
        self,
        method: Union[str, Callable[["Flight"], Iterator["Flight"]]],
    ) -> Optional["Flight"]:
        """
        Returns the final (last) segment of trajectory yielded by
        flight.method()

        >>> flight.final("go_around")
        >>> flight.final("runway_change")
        >>> flight.final(lambda f: f.aligned_on_ils("LFBO"))
        """
        fun = (
            getattr(self.__class__, method)
            if isinstance(method, str)
            else method
        )
        segment = None
        for segment in fun(self):  # noqa: B007
            continue
        return segment  # type: ignore

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

    def coords4d(self, delta_t: bool = False) -> Iterator[Entry]:
        data = self.data.query("longitude == longitude")
        if delta_t:
            time = (data.timestamp - data.timestamp.min()).dt.total_seconds()
        else:
            time = data["timestamp"]

        for t, longitude, latitude, altitude in zip(
            time, data["longitude"], data["latitude"], data["altitude"]
        ):
            if delta_t:
                yield {
                    "timedelta": t,
                    "longitude": longitude,
                    "latitude": latitude,
                    "altitude": altitude,
                }
            else:
                yield {
                    "timestamp": t,
                    "longitude": longitude,
                    "latitude": latitude,
                    "altitude": altitude,
                }

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

    def min(self, feature: str) -> Any:
        """Returns the minimum value of given feature.

        >>> flight.min('altitude')  # dummy example
        24000
        """
        return self.data[feature].min()

    def max(self, feature: str) -> Any:
        """Returns the maximum value of given feature.

        >>> flight.max('altitude')  # dummy example
        35000
        """
        return self.data[feature].max()

    def mean(self, feature: str) -> Any:
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
            return attribute > value  # type: ignore
        return attribute >= value  # type: ignore

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
            return attribute < value  # type: ignore
        return attribute <= value  # type: ignore

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
        """Returns True if flight duration is longer than value."""
        if isinstance(value, str):
            value = pd.Timedelta(value)
        return self.feature_gt(attrgetter("duration"), value, strict)

    def abs(self, features: Union[str, List[str]], **kwargs: Any) -> "Flight":
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

    def diff(self, features: Union[str, List[str]], **kwargs: Any) -> "Flight":
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
            return tmp[0]  # type: ignore
        if warn:
            _log.warning(f"Several {field}s for one flight, consider splitting")
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
    def trip(self) -> str:
        return (
            (
                "("
                if self.origin is not None or self.destination is not None
                else ""
            )
            + (f"{self.origin}" if self.origin else " ")
            + (
                " to "
                if self.origin is not None or self.destination is not None
                else ""
            )
            + (f"{self.destination}" if self.destination else " ")
            + (
                ")"
                if self.origin is not None or self.destination is not None
                else ""
            )
        )

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

        reg = self._get_unique("registration")
        if isinstance(reg, str):
            return reg

        if not isinstance(self.icao24, str):
            return None
        res = aircraft.get_unique(self.icao24)
        if res is None:
            return None
        return res.get("registration", None)

    @property
    def typecode(self) -> Optional[str]:
        from ..data import aircraft

        tc = self._get_unique("typecode")
        if isinstance(tc, str):
            return tc

        if not isinstance(self.icao24, str):
            return None
        res = aircraft.get_unique(self.icao24)
        if res is None:
            return None
        return res.get("typecode", None)

    @property
    def aircraft(self) -> None | Tail:
        from ..data import aircraft

        if isinstance(self.icao24, str):
            return aircraft.get_unique(self.icao24)

        return None

    # -- Time handling, splitting, interpolation and resampling --

    def skip(
        self, value: deltalike = None, **kwargs: Any
    ) -> Optional["Flight"]:
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

    def first(
        self, value: deltalike = None, **kwargs: Any
    ) -> Optional["Flight"]:
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

    def shorten(
        self, value: deltalike = None, **kwargs: Any
    ) -> Optional["Flight"]:
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

    def last(
        self, value: deltalike = None, **kwargs: Any
    ) -> Optional["Flight"]:
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
            _log.warning(f"No index {index} for flight {id_}")
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
        self,
        duration: deltalike,
        step: deltalike,
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
        """Iterates on legs of a Flight based on the distribution of timestamps.

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
        self,
        fun: Callable[..., "LazyTraffic"],
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Optional["Flight"]:
        return getattr(self, name)(*args, **kwargs)(fun)  # type: ignore

    def apply_time(
        self,
        freq: str = "1T",
        merge: bool = True,
        **kwargs: Any,
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
        self,
        freq: str = "1T",
        merge: bool = True,
        **kwargs: Any,
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
            data: pd.DataFrame, how: Callable[..., Any] = "_".join
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

        if not merge:
            # WARN: Return type is inconsistent but this is mostly for
            # debugging purposes
            return agg_data  # type: ignore

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

    def resample(
        self,
        rule: str | int = "1s",
        how: str | dict[str, Iterable[str]] = "interpolate",
        projection: None | str | pyproj.Proj | "crs.Projection" = None,
    ) -> "Flight":
        """Resample the trajectory at a given frequency or for a target number
        of samples.

        :param rule:

            - If the rule is a string representing
              :ref:`pandas:timeseries.offset_aliases` for time frequencies is
              passed, then the data is resampled along the timestamp axis, then
              interpolated (according to the ``how`` parameter).

            - If the rule is an integer, the trajectory is resampled to the
              given number of evenly distributed points per trajectory.

        :param how: (default: ``"interpolate"``)

            - When the parameter is a string, the method applies to all columns
            - When the parameter is a dictionary with keys as methods (e.g.
              ``"interpolate"``, ``"ffill"``) and names of columns as values.
              Columns not included in any value are left as is.

        :param projection: (default: ``None``)

            - By default, lat/lon are resampled with a linear interpolation;
            - If a projection is passed, the linear interpolation is applied on
              the x and y dimensions, then lat/lon are reprojected back;
            - If the projection is a string parameter, e.g. ``"lcc"``, a
              projection is created on the fly, centred on the trajectory. This
              approach is helpful to fill gaps along a great circle.

        """
        if projection is not None:
            if isinstance(projection, str):
                projection = pyproj.Proj(
                    proj=projection,
                    ellps="WGS84",
                    lat_1=self.data.latitude.min(),
                    lat_2=self.data.latitude.max(),
                    lat_0=self.data.latitude.mean(),
                    lon_0=self.data.longitude.mean(),
                )
            self = self.compute_xy(projection=projection)

        if isinstance(rule, str):
            data = (
                self.handle_last_position()
                .unwrap()
                .data.set_index("timestamp")
                .resample(rule)
                .first()
                .reset_index()
            )

            if isinstance(how, str):
                how = {how: set(data.columns) - {"timestamp"}}

            for meth, columns in how.items():
                # final fillna() is necessary for non-interpolable dtypes
                data.loc[:, list(columns)] = getattr(
                    data.loc[:, list(columns)], meth
                )().fillna(method="pad")

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
                .assign(tz_naive=lambda d: d.timestamp.dt.tz_localize(None))
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

        res = self.__class__(data)

        if projection is not None:
            res = res.compute_latlon_from_xy(projection=projection)

        return res

    def filter(
        self,
        strategy: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = lambda x: x.bfill().ffill(),
        **kwargs: int,
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

        import scipy.signal

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

            def identity(x: pd.DataFrame) -> pd.DataFrame:
                return x  # noqa: E704

            strategy = identity

        def cascaded_filters(
            df: pd.DataFrame,
            feature: str,
            kernel_size: int,
            filt: Callable[[pd.Series, int], Any] = scipy.signal.medfilt,
        ) -> pd.DataFrame:
            """Produces a mask for data to be discarded.

            The filtering applies a low pass filter (e.g medfilt) to a signal
            and measures the difference between the raw and the filtered signal.

            The average of the squared differences is then produced (sq_eps) and
            used as a threshold for filtering.

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
        # TODO improve based on agg_time or EKF
        flight: Optional["Flight"] = self
        for _ in range(cascades):
            if flight is None:
                return None
            flight = flight.cumulative_distance().query(
                "compute_gs < compute_gs.mean() + 3 * compute_gs.std()"
            )
        return flight

    def comet(self, **kwargs: Any) -> "Flight":
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
            series = reset.data[feature].astype(float)
            idx = ~series.isnull()
            result_dict[f"{feature}_unwrapped"] = pd.Series(
                np.degrees(np.unwrap(np.radians(series.loc[idx]))),
                index=series.loc[idx].index,
            )

        return reset.assign(**result_dict)

    def compute_TAS(self) -> "Flight":
        """Computes the wind triangle for each timestamp.

        This method requires ``groundspeed``, ``track``, ``wind_u`` and
        ``wind_v`` (in knots) to compute true airspeed (``TAS``), and
        ``heading`` features. The groundspeed and the track angle are usually
        available in ADS-B messages; wind information may be included from a
        GRIB file using the :meth:`~traffic.core.Flight.include_grib` method.

        """

        if any(w not in self.data.columns for w in ["wind_u", "wind_v"]):
            raise RuntimeError(
                "No wind data in trajectory. Consider Flight.include_grib()"
            )

        return self.assign(
            tas_x=lambda df: df.groundspeed * np.sin(np.radians(df.track))
            - df.wind_u,
            tas_y=lambda df: df.groundspeed * np.cos(np.radians(df.track))
            - df.wind_v,
            TAS=lambda df: np.abs(df.tas_x + 1j * df.tas_y),
            heading_rad=lambda df: np.angle(df.tas_x + 1j * df.tas_y),
            heading=lambda df: (90 - np.degrees(df.heading_rad)) % 360,
        ).drop(columns=["tas_x", "tas_y", "heading_rad"])

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
        ax: "GeoAxesSubplot",
        resolution: Union[int, str, Dict[str, float], None] = "5T",
        filtered: bool = False,
        **kwargs: Any,
    ) -> List["Artist"]:  # coverage: ignore
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

            from cartes.crs import Mercator
            fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator()))
            (
                flight
                .resample("1s")
                .query('altitude > 10000')
                .compute_wind()
                .plot_wind(ax, alpha=.5)
            )

        """

        from cartopy.crs import PlateCarree

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

        return ax.barbs(  # type: ignore
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
        size = self.data.shape[0]
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
            size = self.data.shape[0]
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
                        ).geoms
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

    def compute_DME_NSE(
        self,
        dme: "Navaids" | Tuple["Navaid", "Navaid"],
        column_name: str = "NSE",
    ) -> "Flight":
        """Adds the DME/DME Navigation System Error.

        Computes the max Navigation System Error using DME-DME navigation. The
        obtained NSE value corresponds to the 2 :math:`\\sigma` (95%)
        requirement in nautical miles.

        Source: EUROCONTROL Guidelines for RNAV 1 Infrastructure Assessment

        :param dme:

            - when the parameter is of type Navaids, only the pair of Navaid
              giving the smallest NSE are used;
            - when the parameter is of type tuple, the NSE is computed using
              only the pair of specified Navaid.

        :param column_name: (default: ``"NSE"``), the name of the new column
            containing the computed NSE

        """

        from ..data.basic.navaid import Navaids

        sigma_dme_1_sis = sigma_dme_2_sis = 0.05

        def sigma_air(df: pd.DataFrame, column_name: str) -> Any:
            values = df[column_name] * 0.125 / 100
            return np.where(values < 0.085, 0.085, values)

        def angle_from_bearings_deg(
            bearing_1: float, bearing_2: float
        ) -> float:
            # Returns the subtended given by 2 bearings.
            angle = np.abs(bearing_1 - bearing_2)
            return np.where(angle > 180, 360 - angle, angle)  # type: ignore

        if isinstance(dme, Navaids):
            flight = reduce(
                lambda flight, dme_pair: flight.compute_DME_NSE(
                    dme_pair, f"nse_{dme_pair[0].name}_{dme_pair[1].name}"
                ),
                combinations(dme, 2),
                self,
            )
            nse_colnames = list(
                column
                for column in flight.data.columns
                if column.startswith("nse_")
            )
            return (
                flight.assign(
                    NSE=lambda df: df[nse_colnames].min(axis=1),
                    NSE_idx=lambda df: df[nse_colnames].idxmin(axis=1).str[4:],
                )
                .rename(
                    columns=dict(
                        NSE=column_name,
                        NSE_idx=f"{column_name}_idx",
                    )
                )
                .drop(columns=nse_colnames)
            )

        dme1, dme2 = dme
        extra_cols = [
            "b1",
            "b2",
            "d1",
            "d2",
            "sigma_dme_1_air",
            "sigma_dme_2_air",
            "angle",
        ]

        return (
            self.distance(dme1, "d1")
            .bearing(dme1, "b1")
            .distance(dme2, "d2")
            .bearing(dme2, "b2")
            .assign(angle=lambda df: angle_from_bearings_deg(df.b1, df.b2))
            .assign(
                angle=lambda df: np.where(
                    (df.angle >= 30) & (df.angle <= 150), df.angle, np.nan
                )
            )
            .assign(
                sigma_dme_1_air=lambda df: sigma_air(df, "d1"),
                sigma_dme_2_air=lambda df: sigma_air(df, "d2"),
                NSE=lambda df: (
                    2
                    * np.sqrt(
                        df.sigma_dme_1_air**2
                        + df.sigma_dme_2_air**2
                        + sigma_dme_1_sis**2
                        + sigma_dme_2_sis**2
                    )
                )
                / np.sin(np.deg2rad(df.angle)),
            )
            .drop(columns=extra_cols)
            .rename(columns=dict(NSE=column_name))
        )

    def cumulative_distance(
        self,
        compute_gs: bool = True,
        compute_track: bool = True,
        *,
        reverse: bool = False,
        **kwargs: Any,
    ) -> "Flight":

        """Enrich the structure with new ``cumdist`` column computed from
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
            (delta_1.latitude - delta_1.latitude_1).values,
            (delta_1.longitude - delta_1.longitude_1).values,
            delta_1.latitude.values,
            delta_1.longitude.values,
        )

        res = cur_sorted.assign(
            cumdist=np.pad(d.cumsum() / 1852, (1, 0), "constant")
        )

        if compute_gs:
            gs = d / delta_1.timestamp_1.dt.total_seconds() * (3600 / 1852)
            res = res.assign(compute_gs=np.abs(np.pad(gs, (1, 0), "edge")))

        if compute_track:
            track = geo.bearing(
                (delta_1.latitude - delta_1.latitude_1).values,
                (delta_1.longitude - delta_1.longitude_1).values,
                delta_1.latitude.values,
                delta_1.longitude.values,
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
    ) -> npt.NDArray[np.float64] | "Flight":
        """Simplifies a trajectory with Douglas-Peucker algorithm.

        The method uses latitude and longitude, projects the trajectory to a
        conformal projection and applies the algorithm. If x and y features are
        already present in the DataFrame (after a call to
        :ref:`~traffic.core.Flight.compute_xy()` for instance) then this
        projection is taken into account.

        The tolerance parameter must be defined in meters.

        - By default, a 2D version of the algorithm is called, unless you pass a
          column name for ``altitude``.
        - You may scale the z-axis for more relevance (``z_factor``). The
          default value works well in most situations.

        The method returns a :class:`~traffic.core.Flight` or a 1D mask if you
        specify ``return_mask=True``.

        **See also**: :ref:`How to simplify or resample a trajectory?`

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

    @flight_iterator
    def clip_iterate(
        self, shape: Union[ShapelyMixin, base.BaseGeometry], strict: bool = True
    ) -> Iterator["Flight"]:
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
            between = self.between(
                min(time_list), max(time_list), strict=strict
            )
            if between is not None:
                yield between
            return None

        def _clip_generator() -> Iterable[Tuple[datetime, datetime]]:
            for segment in intersection.geoms:
                times: List[datetime] = list(
                    datetime.fromtimestamp(t, timezone.utc)
                    for t in np.stack(segment.coords)[:, 2]
                )
                yield min(times), max(times)

        # it is actually not so simple because of self intersecting trajectories
        prev_t1, prev_t2 = None, None

        for t1, t2 in _clip_generator():
            if prev_t2 is not None and t1 > prev_t2:
                between = self.between(prev_t1, prev_t2, strict=strict)
                if between is not None:
                    yield between
                prev_t1, prev_t2 = t1, t2
            elif prev_t2 is None:
                prev_t1, prev_t2 = t1, t2
            else:
                prev_t1, prev_t2 = min(prev_t1, t1), max(prev_t2, t2)

        if prev_t2 is not None:
            between = self.between(prev_t1, prev_t2, strict=strict)
            if between is not None:
                yield between

    def clip(
        self, shape: Union[ShapelyMixin, base.BaseGeometry], strict: bool = True
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

        t1 = None
        for segment in self.clip_iterate(shape, strict=strict):
            if t1 is None:
                t1 = segment.start
            t2 = segment.stop

        if t1 is None:
            return None

        clipped_flight = self.between(t1, t2, strict=strict)

        if clipped_flight is None or clipped_flight.shape is None:
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

    def query_opensky(self, **kwargs: Any) -> Optional["Flight"]:
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
        return cast(Optional["Flight"], opensky.history(**query_params))

    def query_ehs(
        self,
        data: Union[None, pd.DataFrame, "RawData"] = None,
        failure_mode: str = "warning",
        progressbar: Union[bool, ProgressbarType[Any]] = True,
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

        def fail_warning() -> "Flight":
            """Called when nothing can be added to data."""
            id_ = self.flight_id
            if id_ is None:
                id_ = self.callsign
            _log.warning(f"No data on Impala for flight {id_}.")
            return self

        def fail_silent() -> "Flight":
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

    def leaflet(self, **kwargs: Any) -> Optional[LeafletPolyline]:
        """Returns a Leaflet layer to be directly added to a Map.

        The elements passed as kwargs as passed as is to the PolyLine
        constructor.

        Example usage:

        >>> from ipyleaflet import Map
        >>> # Center the map near the landing airport
        >>> m = Map(center=flight.at().latlon, zoom=7)
        >>> m.add_layer(flight)  # this works as well with default options
        >>> m.add_layer(flight.leaflet(color='red'))
        >>> m
        """
        shape = self.shape
        if shape is None:
            return None

        kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
        return LeafletPolyline(
            locations=list((lat, lon) for (lon, lat, _) in shape.coords),
            **kwargs,
        )

    def map_leaflet(
        self,
        *,
        zoom: int = 7,
        highlight: Optional[
            Dict[
                str,
                Union[str, Flight, Callable[[Flight], Optional[Flight]]],
            ]
        ] = None,
        airport: Union[None, str, Airport] = None,
        **kwargs: Any,
    ) -> Optional[LeafletMap]:
        from ..data import airports

        last_position = self.query("latitude == latitude").at()  # type: ignore
        if last_position is None:
            return None

        _airport = airports[airport] if isinstance(airport, str) else airport

        if "center" not in kwargs:
            if _airport is not None:
                kwargs["center"] = _airport.latlon
            else:
                kwargs["center"] = (
                    self.data.latitude.mean(),
                    self.data.longitude.mean(),
                )

        m = LeafletMap(zoom=zoom, **kwargs)

        if _airport is not None:
            m.add_layer(_airport)

        elt = m.add_layer(self)
        elt.popup = HTML()
        elt.popup.value = self._info_html()

        if highlight is None:
            highlight = dict()

        for color, value in highlight.items():
            if isinstance(value, str):
                value = getattr(Flight, value, None)  # type: ignore
                if value is None:
                    continue
            assert not isinstance(value, str)
            f: Optional[Flight]
            if isinstance(value, Flight):
                f = value
            else:
                f = value(self)
            if f is not None:
                m.add_layer(f, color=color)

        return m

    def plot(
        self, ax: "GeoAxesSubplot", **kwargs: Any
    ) -> List["Artist"]:  # coverage: ignore
        """Plots the trajectory on a Matplotlib axis.

        The Flight supports Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        Example usage:

        .. code:: python

            from cartes.crs import Mercator
            fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator())
            flight.plot(ax, alpha=.5)

        .. note::

            See also `geoencode() <#traffic.core.Flight.geoencode>`_ for the
            altair equivalent.

        """

        from cartopy.crs import PlateCarree

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            kwargs["transform"] = PlateCarree()
        if self.shape is not None:
            return ax.plot(*self.shape.xy, **kwargs)  # type: ignore
        return []

    def chart(self, *features: str) -> "alt.Chart":  # coverage: ignore
        """
        Initializes an altair Chart based on Flight data.

        The features passed in parameters are dispatched to allow plotting
        multiple features on the same graph.

        Example usage:

        .. code:: python

            # Most simple usage
            flight.chart().encode(alt.Y("altitude"))

            # With some configuration
            flight.chart().encode(
                alt.X(
                    "utcyearmonthdatehoursminutes(timestamp)",
                    axis=alt.Axis(title=None, format="%H:%M"),
                ),
                alt.Y("altitude", title="altitude (in ft)"),
                alt.Color("callsign")
            )

        For a more complex graph plotting similar physical quantities on the
        same graph, and other quantities on a different graph, the following
        snippet may be of use.

        .. code:: python

            # More advanced with several plots on the same graph
            base = (
                flight.chart("altitude", "groundspeed", "IAS")
                .encode(
                    alt.X(
                        "utcyearmonthdatehoursminutesseconds(timestamp)",
                        axis=alt.Axis(title=None, format="%H:%M"),
                    )
                )
                .properties(height=200)
            )

            alt.vconcat(
                base.transform_filter('datum.variable != "altitude"').encode(
                    alt.Y(
                        "value:Q",
                        axis=alt.Axis(title="speed (in kts)"),
                        scale=alt.Scale(zero=False),
                    )
                ),
                base.transform_filter('datum.variable == "altitude"').encode(
                    alt.Y("value:Q", title="altitude (in ft)")
                ),
            )

        .. note::

            See also `plot_time() <#traffic.core.Flight.plot_time>`_ for the
            Matplotlib equivalent.

        """
        import altair as alt

        base = alt.Chart(self.data).encode(
            alt.X("utcyearmonthdatehoursminutesseconds(timestamp)"),
        )
        if len(features) > 0:
            base = base.transform_fold(
                list(features), as_=["variable", "value"]
            ).encode(alt.Y("value:Q"), alt.Color("variable:N"))

        return base.mark_line()

    def encode(self, **kwargs: Any) -> NoReturn:  # coverage: ignore
        """
        DEPRECATED: Use Flight.chart() method instead.
        """
        raise DeprecationWarning("Use Flight.chart() method instead")

    def plot_time(
        self,
        ax: "Axes",
        y: Union[str, List[str]],
        secondary_y: Union[None, str, List[str]] = None,
        **kwargs: Any,
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

            See also `chart() <#traffic.core.Flight.chart>`_ for the altair
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

    @classmethod
    def from_file(
        cls: Type[T], filename: Union[Path, str], **kwargs: Any
    ) -> Optional[T]:

        """Read data from various formats.

        This class method dispatches the loading of data in various format to
        the proper ``pandas.read_*`` method based on the extension of the
        filename.

        - .pkl and .pkl.gz dispatch to ``pandas.read_pickle``;
        - .parquet and .parquet.gz dispatch to ``pandas.read_parquet``;
        - .json and .json.gz dispatch to ``pandas.read_json``;
        - .csv and .csv.gz dispatch to ``pandas.read_csv``;
        - .h5 dispatch to ``pandas.read_hdf``.

        Other extensions return ``None``.
        Specific arguments may be passed to the underlying ``pandas.read_*``
        method with the kwargs argument.

        Example usage:

        >>> from traffic.core import Flight
        >>> t = Flight.from_file("example_flight.csv")
        """

        tentative = super().from_file(filename, **kwargs)
        if tentative is None:
            return None

        # Special treatment for flights to download from flightradar24
        cols_fr24 = {
            "Altitude",
            "Callsign",
            "Direction",
            "Position",
            "Speed",
            "Timestamp",
            "UTC",
        }
        if set(tentative.data.columns) != cols_fr24:
            return tentative

        latlon = tentative.data.Position.str.split(pat=",", expand=True)
        return (
            tentative.assign(
                latitude=latlon[0].astype(float),
                longitude=latlon[1].astype(float),
                timestamp=lambda df: pd.to_datetime(df.UTC),
            )
            .rename(
                columns={
                    "UTC": "timestamp",
                    "Altitude": "altitude",
                    "Callsign": "callsign",
                    "Speed": "groundspeed",
                    "Direction": "track",
                }
            )
            .drop(columns=["Timestamp", "Position"])
        )
