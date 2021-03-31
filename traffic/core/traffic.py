# fmt: off

import logging
import warnings
from datetime import timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator,
    List, Optional, Set, Type, TypeVar, Union, overload
)

import pandas as pd
import pyproj
from cartopy import crs
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.artist import Artist
from shapely.geometry import Polygon, base

from ..algorithms.clustering import Clustering, centroid
from ..algorithms.cpa import closest_point_of_approach
from ..core.cache import property_cache
from ..core.time import time_or_delta, timelike, to_datetime
from .flight import Flight, attrgetter_duration
from .lazy import LazyTraffic, lazy_evaluation
from .mixins import GeographyMixin, HBoxMixin, PointMixin
from .sv import StateVectors

if TYPE_CHECKING:
    from .airspace import Airspace  # noqa: F401
    from ..algorithms.cpa import CPA  # noqa: F401
    from ..algorithms.clustering import (  # noqa: F401
        ClusteringProtocol, TransformerProtocol
    )

# fmt: on

# https://github.com/python/mypy/issues/2511
TrafficTypeVar = TypeVar("TrafficTypeVar", bound="Traffic")

# The thing is that Iterable[str] causes issue sometimes...
IterStr = Union[List[str], Set[str]]


class Traffic(HBoxMixin, GeographyMixin):
    """

    Traffic is the abstraction representing a collection of `Flights
    <traffic.core.flight.html>`_. When Flight objects are summed up, the
    resulting structure is a Traffic.

    Data is all flattened into one single pandas DataFrame and methods are
    provided to properly access (with the bracket notation) and iterate on each
    Flight in the structure.

    On top of basic methods and properties (`aircraft
    <traffic.core.Traffic.aircraft>`_, `callsigns
    <traffic.core.Traffic.callsigns>`_, `flight_ids
    <traffic.core.Traffic.flight_ids>`_, `start_time
    <traffic.core.Traffic.start_time>`_, `end_time
    <traffic.core.Traffic.end_time>`_) and data preprocessing (most methods
    available on Flight), more complex algorithms like `closest point of
    approach <#traffic.core.Traffic.closest_point_of_approach>`_ and
    `clustering <#traffic.core.Traffic.clustering>`_ (more to come) are
    available.

    .. note::

        When methods need to be chained on each trajectory contained in the
        collection, **lazy iteration and evaluation** is in place. This means
        that applying such a method on a Traffic structure will only stack
        operations without evaluating them.

        .. autoclass:: traffic.core.lazy.LazyTraffic()
            :members: eval

    .. tip::
        Sample traffic structures are provided for testing purposes in module
        ``traffic.data.samples``

    """

    __slots__ = ("data",)

    @classmethod
    def from_flights(
        cls, flights: Iterable[Optional[Flight]]
    ) -> Optional["Traffic"]:
        """
        Creates a Traffic structure from all flights passed as an
        iterator or iterable.
        """
        cumul = [flight.data for flight in flights if flight is not None]
        if len(cumul) == 0:
            return None
        return cls(pd.concat(cumul, sort=False))

    @classmethod
    def from_file(
        cls: Type[TrafficTypeVar], filename: Union[Path, str], **kwargs
    ) -> Optional[TrafficTypeVar]:

        tentative = super().from_file(filename, **kwargs)

        if tentative is not None:
            rename_columns = {
                "time": "timestamp",
                "lat": "latitude",
                "lon": "longitude",
                "lng": "longitude",
                "long": "longitude",
                # speeds
                "velocity": "groundspeed",
                "ground_speed": "groundspeed",
                "ias": "IAS",
                "tas": "TAS",
                "mach": "Mach",
                # vertical rate
                "vertrate": "vertical_rate",
                "vertical_speed": "vertical_rate",
                "roc": "vertical_rate",
                # let's just make baroaltitude the altitude by default
                "baro_altitude": "altitude",
                "baroaltitude": "altitude",
                "geo_altitude": "geoaltitude",
                # synonyms
                "departure": "origin",
                "arrival": "destination",
            }

            if (
                "baroaltitude" in tentative.data.columns
                or "baro_altitude" in tentative.data.columns
            ):
                # for retrocompatibility
                rename_columns["altitude"] = "geoaltitude"

            if (
                "heading" in tentative.data.columns
                and "track" not in tentative.data.columns
            ):
                # that's a common confusion in data, let's assume that
                rename_columns["heading"] = "track"

            return tentative.rename(columns=rename_columns)

        path = Path(filename)
        logging.warning(f"{path.suffixes} extension is not supported")
        return None

    # --- Special methods ---
    # operators + (union), & (intersection), - (difference), ^ (xor)

    def __add__(self, other) -> "Traffic":
        # useful for compatibility with sum() function
        if other == 0:
            return self
        return self.__class__(pd.concat([self.data, other.data], sort=False))

    def __radd__(self, other) -> "Traffic":
        return self + other

    def __and__(self, other: "Traffic") -> Optional["Traffic"]:
        if not isinstance(other, Traffic):
            raise RuntimeError(
                "Operator `&` is only applicable between Traffic structures."
            )
        list_id = other.flight_ids
        if list_id is None or self.flight_ids is None:
            raise RuntimeError(
                "No flight_id is provided in the given Traffic structures."
            )
        df = self.data.query("flight_id in @list_id")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def __sub__(self, other: "Traffic") -> Optional["Traffic"]:
        # set difference based on flight_id
        if not isinstance(other, Traffic):
            raise RuntimeError(
                "Operator `-` is only applicable between Traffic structures."
            )
        list_id = other.flight_ids
        if list_id is None or self.flight_ids is None:
            raise RuntimeError(
                "No flight_id is provided in the given Traffic structures."
            )
        df = self.data.query("flight_id not in @list_id")
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def __xor__(self, other: "Traffic") -> Optional["Traffic"]:
        left = self - other
        right = other - self
        if left is None:
            return right
        if right is None:
            return left
        return right + left

    def _getSeries(self, index: pd.Series) -> Optional[Flight]:

        p_callsign = hasattr(index, "callsign")
        p_icao24 = hasattr(index, "icao24")

        if p_callsign or p_icao24:
            query = []
            if p_callsign:
                query.append(f"callsign == '{index.callsign}'")
            if p_icao24:
                query.append(f"icao24 == '{index.icao24}'")

            df = self.data.query(
                query[0] if len(query) == 1 else " and ".join(query)
            )
            if df.shape[0] == 0:
                return None

            flight: Optional[Flight] = Flight(df)

            if flight is not None and hasattr(index, "firstSeen"):
                # refers to OpenSky REST API
                flight = flight.after(index.firstSeen)
            if flight is not None and hasattr(index, "lastSeen"):
                # refers to OpenSky REST API
                flight = flight.before(index.lastSeen)

            if flight is not None and hasattr(index, "start"):  # more natural
                flight = flight.after(index.start)
            if flight is not None and hasattr(index, "stop"):  # more natural
                flight = flight.before(index.stop)

            return flight

        return None

    @overload
    def __getitem__(self, index: int) -> Optional[Flight]:
        ...

    @overload
    def __getitem__(self, index: str) -> Optional[Flight]:  # noqa: F811
        ...

    @overload
    def __getitem__(self, index: slice) -> Optional["Traffic"]:  # noqa: F811
        ...

    @overload
    def __getitem__(self, index: IterStr) -> Optional["Traffic"]:  # noqa: F811
        ...

    def __getitem__(  # noqa: F811
        self, index
    ) -> Union[None, Flight, "Traffic"]:

        if isinstance(index, pd.Series):
            return self._getSeries(index)

        if isinstance(index, pd.DataFrame):
            return self.__class__.from_flights(
                flight for flight in self.iterate(by=index)
            )

        if isinstance(index, int):
            for i, flight in enumerate(self.iterate()):
                if i == index:
                    return flight
            return None

        if isinstance(index, slice):
            max_size = index.stop if index.stop is not None else len(self)
            indices = list(range(max_size)[index])
            return self.__class__.from_flights(
                flight
                for i, flight in enumerate(self.iterate())
                if i in indices
            )

        if not isinstance(index, str):  # List[str], Set[str], Iterable[str]
            logging.debug("Selecting flights from a list of identifiers")
            subset = repr(list(index))
            query_str = f"callsign in {subset} or icao24 in {subset}"
            if "flight_id" in self.data.columns:
                return self.query(f"flight_id in {subset} or " + query_str)
            else:
                return self.query(query_str)

        query_str = f"callsign == '{index}' or icao24 == '{index}'"
        if "flight_id" in self.data.columns:
            df = self.data.query(f"flight_id == '{index}' or " + query_str)
        else:
            df = self.data.query(query_str)

        if df.shape[0] > 0:
            return Flight(df)

        return None

    def _ipython_key_completions_(self) -> Set[str]:
        if self.flight_ids is not None:
            return self.flight_ids
        return {*self.aircraft, *self.callsigns}

    def iterate(
        self,
        by: Union[str, pd.DataFrame, None] = None,
        nb_flights: Optional[int] = None,
    ) -> Iterator[Flight]:
        """
        Iterates over Flights contained in the Traffic structure.

        Default iteration calls this method with default arguments:

        >>> for flight in t:
        ...     pass

        is equivalent to:

        >>> for flight in t.iterate():
        ...     pass

        However the it may be beneficial to specify the `by` parameter:

        - as a pandas DataFrame with callsign and or icao24 columns, it
          defines a subset of Flights to select.
        - as a a string, `by` defines the minimum time range without
          data for a flight.
        """

        if isinstance(by, pd.DataFrame):
            for i, (_, line) in enumerate(by.iterrows()):
                if nb_flights is None or i < nb_flights:
                    flight = self[line]
                    if flight is not None:
                        yield flight
            return

        if "flight_id" in self.data.columns:
            for i, (_, df) in enumerate(self.data.groupby("flight_id")):
                if nb_flights is None or i < nb_flights:
                    yield Flight(df)
        else:
            for i, (_, df) in enumerate(
                self.data.sort_values("timestamp").groupby(
                    ["icao24", "callsign"]
                )
            ):
                if nb_flights is None or i < nb_flights:
                    yield from Flight(df).split(
                        by if by is not None else "10 minutes"
                    )

    def iterate_lazy(
        self,
        iterate_kw: Optional[Dict[str, Any]] = None,
        tqdm_kw: Optional[Dict[str, Any]] = None,
    ) -> LazyTraffic:
        """
        Triggers a lazy iteration on the Traffic structure.

        Default iteration calls this method with default arguments:

        >>> t.filter()

        is equivalent to:

        >>> t.iterate_lazy().filter()

        However the it may be beneficial to specify the `by` parameter:

        - as a pandas DataFrame with callsign and or icao24 columns, it
          defines a subset of Flights to select.
        - as a a string, `by` defines the minimum time range without
          data for a flight.

        You may also select parameters to pass to a tentative tqdm
        progressbar.
        """
        if iterate_kw is None:
            iterate_kw = {}
        if tqdm_kw is None:
            tqdm_kw = {}
        return LazyTraffic(self, [], iterate_kw=iterate_kw, tqdm_kw=tqdm_kw)

    def __iter__(self) -> Iterator[Flight]:
        yield from self.iterate()

    @property_cache
    def length(self):
        ids_ = self.flight_ids
        if ids_ is not None:
            return len(ids_)
        return sum(1 for _ in self)

    def __len__(self):
        return self.length

    def __repr__(self) -> str:
        basic_stats = self.basic_stats
        shape = basic_stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            basic_stats = basic_stats.head(10)
        return basic_stats.__repr__()

    def _repr_html_(self) -> str:
        basic_stats = self.basic_stats
        shape = basic_stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            basic_stats = basic_stats.head(10)
        styler = basic_stats.style.bar(align="mid", color="#5fba7d")
        rep = f"<b>Traffic with {shape} identifiers</b>"
        return rep + styler._repr_html_()

    def aircraft_data(self) -> "Traffic":
        """
        Add registration and aircraft typecode based on the `aircraft database
        <aircraft.html>`_.

        """
        from ..data import aircraft

        return self.merge(
            aircraft.data[["icao24", "registration", "typecode"]]
            .query('typecode != ""')
            .drop_duplicates("icao24"),
            on="icao24",
            how="left",
        )

    # -- Methods for lazy evaluation, delegated to Flight --

    @lazy_evaluation()
    def filter_if(self, *args, **kwargs):
        ...

    @lazy_evaluation()
    def has(self, *args, **kwargs):
        ...

    @lazy_evaluation()
    def all(self, *args, **kwargs):
        ...

    @lazy_evaluation()
    def next(self, *args, **kwargs):
        ...

    @lazy_evaluation()
    def resample(self, rule: Union[str, int] = "1s"):
        ...

    @lazy_evaluation()
    def filter(
        self,
        strategy: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = lambda x: x.bfill().ffill(),
        **kwargs,
    ):
        ...

    @lazy_evaluation()
    def filter_position(self, cascades: int = 2):
        ...

    @lazy_evaluation()
    def unwrap(self, features: Union[None, str, List[str]] = None):
        ...

    @lazy_evaluation(idx_name="idx")
    def assign_id(self, name: str = "{self.callsign}_{idx:>03}", idx: int = 0):
        """Assigns a `flight_id` to trajectories present in the structure.

        The heuristics with iterate on flights based on ``flight_id`` (if the
        feature is present) or of ``icao24``, ``callsign`` and intervals of time
        without recorded data.

        The flight_id is created according to a pattern passed in parameter,
        by default based on the callsign and an incremented index.
        """
        ...

    @lazy_evaluation()
    def clip(self, shape: Union["Airspace", base.BaseGeometry]):
        ...

    @lazy_evaluation()
    def intersects(self, shape: Union["Airspace", base.BaseGeometry]) -> bool:
        ...

    @lazy_evaluation()
    def simplify(
        self,
        tolerance: float,
        altitude: Optional[str] = None,
        z_factor: float = 3.048,
    ):
        ...

    @lazy_evaluation()
    def query_opensky(self):
        ...

    @lazy_evaluation()
    def query_ehs(self, data, failure_mode, propressbar):
        ...

    @lazy_evaluation()
    def first(self, **kwargs):
        ...

    @lazy_evaluation()
    def last(self, **kwargs):
        ...

    @lazy_evaluation()
    def feature_gt(
        self,
        feature: Union[str, Callable[["Flight"], Any]],
        value: Any,
        strict: bool = True,
    ):
        ...

    @lazy_evaluation()
    def feature_lt(
        self,
        feature: Union[str, Callable[["Flight"], Any]],
        value: Any,
        strict: bool = True,
    ):
        ...

    @lazy_evaluation()
    def shorter_than(
        self, value: Union[str, timedelta, pd.Timedelta], strict: bool = True
    ):
        ...

    @lazy_evaluation()
    def longer_than(
        self, value: Union[str, timedelta, pd.Timedelta], strict: bool = True
    ):
        ...

    @lazy_evaluation()
    def max_split(
        self,
        value: Union[int, str] = "10T",
        unit: Optional[str] = None,
        key: Callable[[Optional["Flight"]], Any] = attrgetter_duration,
    ):
        ...

    @lazy_evaluation()
    def apply_segments(
        self, fun: Callable[..., "LazyTraffic"], name: str, *args, **kwargs
    ):
        ...

    @lazy_evaluation()
    def apply_time(self, freq="1T", merge=True, **kwargs):
        ...

    @lazy_evaluation()
    def agg_time(self, freq="1T", merge=True, **kwargs):
        ...

    @lazy_evaluation()
    def cumulative_distance(
        self, compute_gs: bool = True, compute_track: bool = True, **kwargs
    ):
        ...

    @lazy_evaluation()
    def compute_wind(self):
        ...

    @lazy_evaluation()
    def distance(
        self,
        other: Union["Airspace", Polygon, PointMixin],
        column_name: str = "distance",
    ):
        ...

    @lazy_evaluation()
    def landing_at(self, airport: str) -> bool:
        ...

    # -- Methods with a Traffic implementation, otherwise delegated to Flight

    @lazy_evaluation(default=True)
    def before(self, ts: timelike, strict: bool = True) -> Optional["Traffic"]:
        return self.between(self.start_time, ts, strict)

    @lazy_evaluation(default=True)
    def after(self, ts: timelike, strict: bool = True) -> Optional["Traffic"]:
        return self.between(ts, self.end_time, strict)

    @lazy_evaluation(default=True)
    def between(
        self, start: timelike, stop: time_or_delta, strict: bool = True
    ) -> Optional["Traffic"]:

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

        # full call is necessary to keep @before and @after as local variables
        # return self.query('@before < timestamp < @after')  => not valid
        if strict:
            df = self.data.query("@start < timestamp < @stop")
        else:
            df = self.data.query("@start <= timestamp <= @stop")

        if df.shape[0] == 0:
            return None

        return self.__class__(df)

    @lazy_evaluation(default=True)
    def airborne(self) -> Optional["Traffic"]:
        """Returns the airborne part of the Traffic.

        The airborne part is determined by an ``onground`` flag or null values
        in the altitude column.
        """
        if "onground" in self.data.columns and self.data.onground.dtype == bool:
            return self.query("not onground and altitude == altitude")
        else:
            return self.query("altitude == altitude")

    # --- Properties ---

    @property_cache
    def start_time(self) -> pd.Timestamp:
        """Returns the earliest timestamp in the DataFrame."""
        return self.data.timestamp.min()

    @property_cache
    def end_time(self) -> pd.Timestamp:
        """Returns the latest timestamp in the DataFrame."""
        return self.data.timestamp.max()

    @property_cache
    def callsigns(self) -> Set[str]:
        """Return all the different callsigns in the DataFrame"""
        sub = self.data.query("callsign == callsign")
        if sub.shape[0] == 0:
            return set()
        return set(sub.callsign)

    @property_cache
    def aircraft(self) -> Set[str]:
        """Return all the different icao24 aircraft ids in the DataFrame"""
        logging.warning("Use .icao24", DeprecationWarning)
        return set(self.data.icao24)

    @property_cache
    def icao24(self) -> Set[str]:
        """Return all the different icao24 aircraft ids in the DataFrame"""
        return set(self.data.icao24)

    @property_cache
    def flight_ids(self) -> Optional[List[str]]:
        """Return all the different flight_id in the DataFrame"""
        if "flight_id" in self.data.columns:
            return list(flight.flight_id for flight in self)  # type: ignore
        return None

    # --- Easy work ---

    def at(self, time: Optional[timelike] = None) -> "StateVectors":
        if time is not None:
            time = to_datetime(time)
            list_flights = [
                flight.at(time)
                for flight in self
                if flight.start <= time <= flight.stop
            ]
        else:
            list_flights = [flight.at() for flight in self]
        return StateVectors(
            pd.DataFrame.from_records(
                [s for s in list_flights if s is not None]
            ).assign(
                # attribute 'name' refers to the index, i.e. 'timestamp'
                timestamp=[s.name for s in list_flights if s is not None]
            )
        )

    @property_cache
    def basic_stats(self) -> pd.DataFrame:
        key = ["icao24", "callsign"] if self.flight_ids is None else "flight_id"
        return (
            self.data.groupby(key)[["timestamp"]]
            .count()
            .sort_values("timestamp", ascending=False)
            .rename(columns={"timestamp": "count"})
        )

    def summary(
        self,
        attributes: List[str],
        iterate_kw: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Returns a summary of the current Traffic structure containing
        featured attributes.

        Example usage (TODO)

        """
        if iterate_kw is None:
            iterate_kw = dict()
        return pd.DataFrame.from_records(
            dict((key, getattr(flight, key)) for key in attributes)
            for flight in self.iterate(**iterate_kw)
        )

    def geoencode(self, *args, **kwargs):
        """
        .. danger::
            This method is not implemented.
        """
        raise NotImplementedError

    def plot(
        self, ax: GeoAxesSubplot, nb_flights: Optional[int] = None, **kwargs
    ) -> None:  # coverage: ignore
        """Plots each trajectory on a Matplotlib axis.

        Each Flight supports Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        Example usage:

        >>> from traffic.drawing import EuroPP
        >>> fig, ax = plt.subplots(1, subplot_kw=dict(projection=EuroPP()))
        >>> t.plot(ax, alpha=.5)

        """
        params: Dict[str, Any] = {}
        if nb_flights is not None:
            warnings.warn(
                "nb_flights will disappear in future versions. "
                "Use indexing [:nb_flights] before plotting instead",
                DeprecationWarning,
            )
        if sum(1 for _ in zip(range(8), self)) == 8:
            params["color"] = "#aaaaaa"
            params["linewidth"] = 1
            params["alpha"] = 0.8
            kwargs = {**params, **kwargs}  # precedence of kwargs over params
        for i, flight in enumerate(self):
            if nb_flights is None or i < nb_flights:
                flight.plot(ax, **kwargs)

    def agg_latlon(
        self, resolution: Union[Dict[str, float], None] = None, **kwargs
    ) -> pd.DataFrame:
        """Aggregates values of a traffic over a grid of lat/lon.

        The resolution of the grid is passed as a dictionary parameter.
        By default, the grid is made by rounding latitudes and longitudes to
        the nearest integer values. ``dict(latitude=2, longitude=4)``
        will take 2 values per integer latitude intervals (43, 43.5, 44, ...)
        and 4 values per integer longitude intervals (1, 1.25, 1.5, 1.75, ...).

        The kwargs specifies how to aggregate values:

        - ``altitude="mean"`` would average all values in the given cell;
        - ``timestamp="count"`` would return the number of samples per cell;
        - ``icao24="nunique"`` would return the number of different aircraft
          int the given cell.

        The returned pandas DataFrame is indexed over latitude and longitude
        values. It is conveniently chainable with the ``.to_xarray()`` method
        in order to plot density heatmaps.

        Example usage:

        .. code:: python

            switzerland.agg_latlon(
                resolution=dict(latitude=10, longitude=10),
                vertical_rate="mean",
                timestamp="count"
            )

        See how to make `flight density heatmaps </scenarios/heatmaps.html>`_
        """
        warnings.warn(
            "agg_latlon will disappear in future versions. "
            "Use agg_xy instead",
            DeprecationWarning,
        )

        if resolution is None:
            resolution = dict(latitude=1, longitude=1)

        if len(kwargs) is None:
            raise ValueError(
                "Specify parameters to aggregate, "
                "e.g. altitude='mean' or icao24='nunique'"
            )

        r_lat = resolution.get("latitude", None)
        r_lon = resolution.get("longitude", None)

        if r_lat is None or r_lon is None:
            raise ValueError("Specify a resolution for latitude and longitude")

        data = (
            self.assign(
                latitude=lambda x: ((r_lat * x.latitude).round() / r_lat),
                longitude=lambda x: ((r_lon * x.longitude).round() / r_lon),
            )
            .groupby(["latitude", "longitude"])
            .agg(kwargs)
        )
        return data

    def windfield(
        self, resolution: Union[Dict[str, float], None] = None
    ) -> pd.DataFrame:

        if any(w not in self.data.columns for w in ["wind_u", "wind_v"]):
            raise RuntimeError(
                "No wind data in trajectory. Consider Traffic.compute_wind()"
            )

        return self.agg_latlon(
            resolution=resolution,
            wind_u="mean",
            wind_v="mean",
            timestamp="count",
        )

    def plot_wind(
        self,
        ax: GeoAxesSubplot,
        resolution: Union[Dict[str, float], None] = None,
        threshold: int = 10,
        filtered: bool = False,
        **kwargs,
    ) -> List[Artist]:  # coverage: ignore
        """Plots the wind field seen by the aircraft on a Matplotlib axis.

        The Flight supports Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        The `resolution` argument may be:

            - a dictionary, e.g dict(latitude=4, longitude=4), if you
              want a grid with a resolution of 4 points per latitude and
              longitude degree.
            - None (default) for dict(latitude=1, longitude=1)

        Example usage:

        >>> from traffic.drawing import Mercator
        >>> fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator()))
        >>> (
        ...     traffic
        ...     .resample("1s")
        ...     .query('altitude > 10000')
        ...     .compute_wind()
        ...     .eval()
        ...     .plot_wind(ax, alpha=.5)
        ... )

        """

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            kwargs["transform"] = crs.PlateCarree()

        if any(w not in self.data.columns for w in ["wind_u", "wind_v"]):
            raise RuntimeError(
                "No wind data in trajectory. Consider Traffic.compute_wind()"
            )

        data = (
            (
                self.iterate_lazy()
                .filter(roll=17)
                .query("roll.abs() < .5")
                .filter(wind_u=17, wind_v=17)
                .eval(desc="")
            )
            if filtered
            else self
        )

        windfield = (
            data.windfield(resolution)
            .query(f"timestamp > {threshold}")
            .reset_index()
        )

        return ax.barbs(
            windfield.longitude.values,
            windfield.latitude.values,
            windfield.wind_u.values,
            windfield.wind_v.values,
            **kwargs,
        )

    # --- Real work ---

    def clean_invalid(self, threshold: int = 10) -> "Traffic":
        """Removes irrelevant data from the Traffic DataFrame.

        Data that has been downloaded from the OpenSky Impala shell often
        contains faulty data, esp. because of faulty callsigns (wrongly decoded?
        faulty crc?) and of automatically repeated positions (see
        `last_position`).

        This methods is an attempt to automatically clean this data.

        Data uncleaned could result in the following count of messages
        associated to aircraft icao24 `02008b` which could be easily removed.

        .. parsed-literal::
                                   count
            icao24  callsign
            02008b  0  221         8
                    2AM2R1         4
                    2N D           1
                    3DYCI          1
                    3N    I8       1
                    3Q G9 E        1
                    6  V X         1
                    [...]

        """

        if "last_position" not in self.data.columns:
            return self

        return self.__class__(
            self.data.groupby(["icao24", "callsign"]).filter(
                lambda x: x.drop_duplicates("last_position").count().max()
                > threshold
            )
        )

    def closest_point_of_approach(
        self,
        lateral_separation: float,
        vertical_separation: float,
        projection: Union[pyproj.Proj, crs.Projection, None] = None,
        round_t: str = "d",
        max_workers: int = 4,
    ) -> Optional["CPA"]:
        """
        Computes a Closest Point of Approach (CPA) dataframe for all pairs of
        trajectories candidates for being separated by less than
        lateral_separation in vertical_separation.

        The problem of iterating over pairs of trajectories is of unreasonable
        complexity O(n**2). Therefore, instead of computing the CPA between all
        pairs of trajectory, we do it for all pairs of trajectories coming
        closer than a given ``lateral_separation`` and ``vertical_separation``.

        lateral_separation: float (in **meters**)
            Depending on your application, you could start with 10 * 1852 (for
            10 nautical miles)

        vertical_separation: float (in ft)
            Depending on your application, you could start with 1500 (feet)

        projection: pyproj.Proj, crs.Projection, None
            a first filtering is applied on the bounding boxes of trajectories,
            expressed in meters. You need to provide a decent projection able to
            approximate distances by Euclide formula. By default, EuroPP()
            projection is considered, but a non explicit argument will raise a
            warning.

        round_t: str
            an additional column will be added in the DataFrame to group
            trajectories by relevant time frames. Distance computations will be
            considered only between trajectories flown in the same time frame.
            By default, the 'd' pandas freq parameter is considered, to group
            trajectories by day, but other ways of splitting ('h') may be more
            relevant and impact performance.

        max_workers: int
            distance computations are spread over a given number of
            processors.

        Returns a CPA DataFrame wrapper.

        """

        return closest_point_of_approach(
            self,
            lateral_separation,
            vertical_separation,
            projection,
            round_t,
            max_workers,
        )

    def clustering(
        self,
        clustering: "ClusteringProtocol",
        nb_samples: Optional[int],
        features: Optional[List[str]] = None,
        *args,
        projection: Union[None, crs.Projection, pyproj.Proj] = None,
        transform: Optional["TransformerProtocol"] = None,
        max_workers: int = 1,
        return_traffic: bool = True,
    ) -> Clustering:
        """
        Computes a clustering of the trajectories, add labels in a column
        ``cluster``.

        The method:

            - resamples all trajectories with the same number of samples
              ``nb_samples`` (no default value);
            - *if need be,* computes x and y coordinates based on ``projection``
              through a call to `compute_xy()
              <#traffic.core.Traffic.compute_xy>`_ (no default value);
            - *if need be,* apply a transformer to the resulting `X` matrix.
              You may want to consider `StandardScaler()
              <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_;
            - generates the appropriate structure for a call to the usual
              `sklearn API
              <https://scikit-learn.org/stable/modules/clustering.html#clustering>`_
              that is a class with a ``fit()`` method and a ``predict()`` method
              or a ``labels_`` attribute;
            - returns a Clustering object, on which to call fit(), predict() or
              fit_predict() methods. Predicting methods return the original
              Traffic DataFrame with an additional ``cluster`` column.

        Example usage:

        >>> from traffic.core.projection import EuroPP
        >>> from sklearn.cluster import DBSCAN
        >>> from sklearn.preprocessing import StandardScaler
        >>>
        >>> t_dbscan = traffic.clustering(
        ...     nb_samples=15,
        ...     projection=EuroPP(),
        ...     clustering=DBSCAN(eps=1.5, min_samples=10),
        ...     transform=StandardScaler(),
        ... ).fit_predict()
        >>> t_dbscan.groupby(["cluster"]).agg({"flight_id": "nunique"})

        .. parsed-literal::
                        flight_id
            cluster
            -1          15
            0           29
            1           13
            2           24
            3           24

        """

        if features is None:
            features = ["x", "y"]

        return Clustering(
            self,
            clustering,
            nb_samples,
            features,
            projection=projection,
            transform=transform,
        )

    def centroid(
        self,
        nb_samples: Optional[int],
        features: Optional[List[str]] = None,
        projection: Union[None, crs.Projection, pyproj.Proj] = None,
        transformer: Optional["TransformerProtocol"] = None,
        max_workers: int = 1,
        *args,
        **kwargs,
    ) -> "Flight":
        """
        Returns the trajectory in the Traffic that is the closest to all other
        trajectories.

        .. warning::
            Remember the time and space complexity of this method is in O(n^2).

        *args and **kwargs are passed as is to `scipy.spatial.pdist
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist>`_

        """

        if features is None:
            features = ["x", "y"]

        return centroid(
            self,
            nb_samples,
            features,
            projection,
            transformer,
            max_workers,
            *args,
            **kwargs,
        )
