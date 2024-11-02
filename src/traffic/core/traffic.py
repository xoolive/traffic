from __future__ import annotations

import logging
import warnings
from datetime import timedelta
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
    Sequence,
    Set,
    Union,
    cast,
    overload,
)

from rich.console import Console, ConsoleOptions, RenderResult
from typing_extensions import Self

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Polygon, base

from ..algorithms.clustering import Clustering, centroid
from ..algorithms.cpa import closest_point_of_approach
from ..algorithms.generation import Generation
from ..core.cache import property_cache
from ..core.structure import Airport
from ..core.time import time_or_delta, timelike, to_datetime
from .flight import Flight
from .intervals import Interval, IntervalCollection
from .lazy import LazyTraffic, lazy_evaluation
from .mixins import DataFrameMixin, GeographyMixin, HBoxMixin, PointMixin
from .sv import StateVectors

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from cartopy import crs
    from cartopy.mpl.geoaxes import GeoAxesSubplot
    from ipyleaflet import Map as LeafletMap
    from matplotlib.artist import Artist

    from ..algorithms.clustering import ClusteringProtocol, TransformerProtocol
    from ..algorithms.cpa import CPA
    from ..algorithms.generation import GenerationProtocol, ScalerProtocol
    from .airspace import Airspace


# https://github.com/python/mypy/issues/2511
# TrafficTypeVar = TypeVar("TrafficTypeVar", bound="Traffic")

# The thing is that Iterable[str] causes issue sometimes...
IterStr = Union[List[str], Set[str]]

_log = logging.getLogger(__name__)


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
    def from_file(cls, filename: Union[Path, str], **kwargs: Any) -> Self:
        tentative = super().from_file(filename, **kwargs)

        if tentative is not None:
            tentative = tentative.convert_dtypes(dtype_backend="pyarrow")

            for column in tentative.data.select_dtypes(include=["datetime"]):
                if tentative.data[column].dtype.pyarrow_dtype.tz is None:
                    tentative = tentative.assign(
                        **{column: tentative.data[column].dt.tz_localize("UTC")}
                    )

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
        _log.warning(f"{path.suffixes} extension is not supported")
        return None

    @classmethod
    def from_fr24(
        cls,
        metadata: str | Path,
        trajectories: str | Path,
    ) -> Traffic:
        """Parses data as usually provided by FlightRadar24.

        When FlightRadar24 provides data excerpts from their database, they
        usually provide:

        :param metadata: a CSV file with metadata

        :param trajectories: a zip file containing one file per flight with
            trajectory information.

        :return: a regular Traffic object.
        """
        from ..data.datasets.flightradar24 import FlightRadar24

        return FlightRadar24.from_archive(metadata, trajectories)

    # --- Special methods ---
    # operators + (union), & (intersection), - (difference), ^ (xor)

    def __add__(self, other: Literal[0] | Flight | Traffic) -> Traffic:
        """Concatenation operator.

        :param other: is the other Flight or Traffic.

        :return: The sum of two Traffic returns a Traffic collection.
            Summing a Traffic with 0 returns the same Traffic, for compatibility
            reasons with the sum() builtin.

        """
        # useful for compatibility with sum() function
        if other == 0:
            return self
        return self.__class__(pd.concat([self.data, other.data], sort=False))

    def __radd__(
        self, other: Union[Literal[0], Flight, "Traffic"]
    ) -> "Traffic":
        return self + other

    def __and__(self, other: "Traffic") -> Optional["Traffic"]:
        """Intersection of collections.

        :param other:

        :return: the subset of trajectories present in both collections.
            The result is based on the ``flight_id`` if present, and on
            intervals otherwise.
        """
        if not isinstance(other, Traffic):
            return NotImplemented

        # The easy case, when Traffic collections have a ``flight_id``
        if (flight_ids := other.flight_ids) is not None:
            if self.flight_ids is not None:
                df = self.data.query("flight_id in @flight_ids")
                if df.shape[0] == 0:
                    return None
                return self.__class__(df)

        # Otherwise build intervals
        cumul = []
        self_stats = self.summary(["icao24", "start", "stop"]).eval()
        columns = ["icao24", "start", "stop"]
        if flight_ids is not None:
            columns.append("flight_id")
        other_stats = cast(pd.DataFrame, other.summary(columns).eval())
        assert self_stats is not None and other_stats is not None
        for (icao24,), lines in self_stats.groupby(["icao24"]):
            self_interval = IntervalCollection(lines)
            self_interval = self_interval.reset_index()
            other_df = other_stats.query("icao24 == @icao24")
            if other_df.shape[0] > 0:
                other_interval = IntervalCollection(other_df)
                other_interval = other_interval.reset_index()
                if overlap := self_interval & other_interval:
                    for interval in overlap:
                        line = other_df.query(
                            "start <= @interval.start and "
                            "stop >= @interval.stop"
                        )
                        if flight_ids:
                            cumul.append(
                                dict(
                                    start=interval.start,
                                    stop=interval.stop,
                                    icao24=icao24,
                                    flight_id=line.iloc[0].flight_id,
                                )
                            )
                        else:
                            cumul.append(
                                dict(
                                    start=interval.start,
                                    stop=interval.stop,
                                    icao24=icao24,
                                )
                            )

        if len(cumul) == 0:
            return None

        result_intervals = pd.DataFrame.from_records(cumul)
        return self[result_intervals]  # type: ignore

    def __sub__(
        self, other: str | list[str] | set[str] | Flight | Traffic
    ) -> None | Traffic:
        """Remove trajectories from a Traffic object.

        :param other:
            - When the ``other`` attribute is a string, or a list/set of
              strings, all flights matching the ``flight_id``, ``callsign`` or
              ``icao24`` attribute are removed from the collection;
            - When the ``other`` attribute is a Flight, the collection will be
              pruned of the trajectory with the same ``flight_id``; or the
              segment of trajectory for that ``icao24`` address between the
              ``start`` and ``stop`` timestamps;
            - When the ``other`` attribute is a Traffic object, the difference
              is computed based on the ``flight_id`` if both structures have
              one; otherwise, we iterate through flights and consider removing
              part of the trajectory.

        :return: a new collection of trajectories as a Traffic object

        """
        if isinstance(other, str):
            other = [other]

        if isinstance(other, list) or isinstance(other, set):
            if "flight_id" in self.data.columns:
                df = self.data.query(
                    "flight_id not in @other and"
                    "icao24 not in @other and callsign not in @other"
                )
            elif "callsign" in self.data.columns:
                df = self.data.query(
                    "icao24 not in @other and callsign not in @other"
                )
            else:
                df = self.data.query("icao24 not in @other")
            if df.shape[0] == 0:
                return None
            return self.__class__(df)

        elif isinstance(other, Flight):
            if (flight_id := other.flight_id) is None:
                # If no flight_id is set, look icao24 and timestamps
                df = self.data.query(
                    "not (icao24 == @other.icao24 and "
                    "timestamp <= @other.stop and timestamp >= @other.start)"
                )
                if df.shape[0] == 0:
                    return None
                return self.__class__(df)
            else:
                df = self.data.query(f"flight_id not in {flight_id!r}")
                if df.shape[0] == 0:
                    return None
                return self.__class__(df)

        elif isinstance(other, Traffic):
            list_id = other.flight_ids
            if list_id is not None and self.flight_ids is not None:
                # Quite direct if we have flight_ids in both Traffic objects
                df = self.data.query("flight_id not in @list_id")
                if df.shape[0] == 0:
                    return None
                return self.__class__(df)
            else:
                # Remove flights one by one if no flight_id exists
                cumul = []
                self_stats = self.summary(["icao24", "start", "stop"]).eval()
                other_stats = other.summary(["icao24", "start", "stop"]).eval()
                assert self_stats is not None and other_stats is not None
                for (icao24,), lines in self_stats.groupby(["icao24"]):
                    self_interval = IntervalCollection(lines)
                    self_interval = self_interval.reset_index()
                    other_df = other_stats.query("icao24 == @icao24")
                    if cast(pd.DataFrame, other_df).shape[0] == 0:
                        for interval in self_interval:
                            cumul.append(
                                dict(
                                    start=interval.start,
                                    stop=interval.stop,
                                    icao24=icao24,
                                )
                            )
                    else:
                        other_interval = IntervalCollection(other_df)
                        other_interval = other_interval.reset_index()
                        if difference := self_interval - other_interval:
                            for interval in difference:
                                cumul.append(
                                    dict(
                                        start=interval.start,
                                        stop=interval.stop,
                                        icao24=icao24,
                                    )
                                )
                if len(cumul) == 0:
                    return None

                result_intervals = pd.DataFrame.from_records(cumul)
                return self[result_intervals]  # type: ignore

        return NotImplemented

    def __xor__(self, other: "Traffic") -> None | Traffic:
        left = self - other
        right = other - self
        if left is None:
            return right
        if right is None:
            return left
        return right + left

    def _getSeries(self, index: pd.Series) -> None | Flight:
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
                flight = flight.after(index.firstSeen, strict=False)
            if flight is not None and hasattr(index, "lastSeen"):
                # refers to OpenSky REST API
                flight = flight.before(index.lastSeen, strict=False)

            if flight is not None and hasattr(index, "start"):  # more natural
                flight = flight.after(index.start, strict=False)
            if flight is not None and hasattr(index, "stop"):  # more natural
                flight = flight.before(index.stop, strict=False)
            if flight is not None and hasattr(index, "flight_id"):
                flight = flight.assign(flight_id=index.flight_id)

            return flight

        return None

    @overload
    def __getitem__(self, key: int) -> Flight: ...

    @overload
    def __getitem__(self, key: str) -> Flight: ...

    @overload
    def __getitem__(self, key: slice) -> Traffic: ...

    @overload
    def __getitem__(self, key: IterStr) -> Traffic: ...

    @overload
    def __getitem__(self, key: Traffic) -> Traffic: ...

    def __getitem__(
        self,
        key: int | slice | str | IterStr | pd.Series | pd.DataFrame | Traffic,
    ) -> Flight | Traffic:
        """Indexation of collections.

        :param key:
            - if the key is an integer, will return a Flight object
              (in order of iteration);
            - if the key is a slice, will return a Traffic object
              (in order of iteration);
            - if the key is a string, will return a Flight object, based on
              the ``flight_id``, ``icao24`` or ``callsign``;
            - if the key is a list of string, will return a Traffic object,
              based on the same criteria as above;
            - if the key is a pd.Series, will return a Flight object.
              The key must contain an ``icao24`` feature. It may contain a
              ``callsign``, a ``start`` (or ``firstSeen``) timestamp, a ``stop``
              (or ``lastSeen``) timestamp. If it contains a ``flight_id``
              column, this will be assigned to the Flight.
            - if the key is a pd.DataFrame, will return a Traffic object.
              The key must contain an ``icao24`` feature. It may contain a
              ``callsign``, a ``start`` (or ``firstSeen``) timestamp, a ``stop``
              (or ``lastSeen``) timestamp. If it contains a ``flight_id``
              column, this will be assigned to the Flight.
            - if the key is a Traffic object, will return a new Traffic
              collection, based on the ``flight_id`` elements present in key.
              If no ``flight_id`` is available, it will return the subset of
              trajectories in ``self`` that overlap with any trajectory in key
              (with the same icao24 indicator)


        :return: According to the type of the key, the result could be
            a Flight or a Traffic object.

        """
        if isinstance(key, pd.Series):
            flight = self._getSeries(key)
            if flight is not None:
                return flight
            raise KeyError(f"Indexing with {key} returns an empty result")

        if isinstance(key, pd.DataFrame):
            traffic = self.__class__.from_flights(
                flight for flight in self.iterate(by=key)
            )
            if traffic is not None:
                return traffic
            raise KeyError(f"Indexing with {key} returns an empty result")

        if isinstance(key, int):
            for i, flight in enumerate(self.iterate()):
                if i == key:
                    return flight
            raise KeyError(f"Indexing with {key} returns an empty result")

        if isinstance(key, slice):
            max_size = key.stop if key.stop is not None else len(self)
            indices = list(range(max_size)[key])
            traffic = self.__class__.from_flights(
                flight
                for i, flight in enumerate(self.iterate())
                if i in indices
            )
            if traffic is not None:
                return traffic
            raise KeyError(f"Indexing with {key} returns an empty result")

        if isinstance(key, Traffic):
            if (flight_ids := key.flight_ids) is not None:
                if self.flight_ids is not None:
                    return self[flight_ids]  # type: ignore

            cumul: list[Flight] = []
            columns = ["icao24", "start", "stop"]
            if flight_ids is not None:
                columns.append("flight_id")
            other_stats = cast(pd.DataFrame, key.summary(columns).eval())
            for flight in self:
                other_df = other_stats.query("icao24 == @flight.icao24")
                if other_df.shape[0] > 0:
                    other_interval = IntervalCollection(other_df)
                    if any(
                        Interval(flight.start, flight.stop).overlap(other)
                        for other in other_interval
                    ):
                        if flight_ids:
                            line = other_df.query(
                                "start <= @flight.stop and "
                                "stop >= @flight.start"
                            )
                            cumul.append(
                                flight.assign(flight_id=line.iloc[0].flight_id)
                            )
                        else:
                            cumul.append(flight)
            traffic = Traffic.from_flights(cumul)
            if traffic is not None:
                return traffic
            raise KeyError(f"Indexing with {key} returns an empty result")

        if not isinstance(key, str):  # List[str], Set[str], Iterable[str]
            _log.debug("Selecting flights from a list of identifiers")
            subset = repr(list(key))
            query_str = f"callsign in {subset} or icao24 in {subset}"
            if "flight_id" in self.data.columns:
                traffic = self.query(f"flight_id in {subset} or " + query_str)
            elif "callsign" in self.data.columns:
                traffic = self.query(query_str)
            else:
                traffic = self.query(f"icao24 in {subset}")
            if traffic is not None:
                return traffic
            raise KeyError(f"Indexing with {key} returns an empty result")

        query_str = f"callsign == '{key}' or icao24 == '{key}'"
        if "callsign" not in self.data.columns:
            query_str = f"icao24 == '{key}'"
        if "flight_id" in self.data.columns:
            df = self.data.query(f"flight_id == '{key}' or " + query_str)
        else:
            df = self.data.query(query_str)

        if df.shape[0] > 0:
            return Flight(df)

        raise KeyError(f"Key '{key}' not found")

    def _ipython_key_completions_(self) -> Optional[List[str]]:
        if self.flight_ids is not None:
            return self.flight_ids  # type: ignore
        return list({*self.icao24, *self.callsigns})

    def sample(
        self,
        n: Optional[int] = None,
    ) -> None | Traffic:
        """Returns a random sample of traffic data.

        :param self: An instance of the Traffic class.
        :param n: An integer specifying the number of samples to take from the
            dataset. Default is None, in which case a single value is returned.

        :return: A Traffic of n random sampled flights.
        """
        rng = np.random.default_rng()

        sampled_ids: list[str] = list(
            rng.choice(self.data.flight_id.unique(), size=n, replace=False)
        )
        return self[sampled_ids]

    def iterate(
        self,
        on: Sequence[str] = ["icao24", "callsign"],
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

        If the callsign shouldn't be used for iteration, you may specify it
        using the `on` keyword argument.

        """
        # this is to avoid modifying the default keyword arg
        on_list: str | List[str] = on if isinstance(on, str) else list(on)
        if "callsign" not in self.data.columns and "callsign" in on_list:
            on_list = "icao24"

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
                self.data.sort_values("timestamp").groupby(on_list)
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
    def length(self) -> int:
        ids_ = self.flight_ids
        if ids_ is not None:
            return len(ids_)
        return sum(1 for _ in self)

    def __len__(self) -> int:
        return self.length  # type: ignore

    def __repr__(self) -> str:
        basic_stats = self.basic_stats.reset_index()
        shape = basic_stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            basic_stats = basic_stats.head(10)
        return repr(basic_stats)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        class Stats(DataFrameMixin):
            pass

        mock = Stats(self.basic_stats.reset_index())
        yield "[bold blue]Traffic"
        yield from mock.__rich_console__(console, options)

    def _repr_html_(self) -> str:
        basic_stats = self.basic_stats
        shape = basic_stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            basic_stats = basic_stats.head(10)
        styler = basic_stats.style.bar(align="mid", color="#5fba7d")
        rep = f"<h4><b>Traffic</b></h4> with {shape} identifiers"
        return rep + styler._repr_html_()  # type: ignore

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
    def filter_if(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def pipe(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def has(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def all(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def next(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def final(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def resample(self, /, rule: Union[str, int] = "1s"):  # type: ignore
        ...

    @lazy_evaluation()
    def filter(self, /, *args, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def filter_position(self, /, cascades: int = 2):  # type: ignore
        ...

    @lazy_evaluation()
    def unwrap(  # type: ignore
        self, /, features: Union[None, str, List[str]] = None
    ): ...

    @lazy_evaluation(idx_name="idx")
    def assign_id(  # type: ignore
        self, /, name: str = "{self.callsign}_{idx:>03}", idx: int = 0
    ):
        """Assigns a `flight_id` to trajectories present in the structure.

        The heuristics with iterate on flights based on ``flight_id`` (if the
        feature is present) or of ``icao24``, ``callsign`` and intervals of time
        without recorded data.

        The flight_id is created according to a pattern passed in parameter,
        by default based on the callsign (if any) and an incremented index.
        """
        ...

    @lazy_evaluation()
    def clip(self, /, shape: Union["Airspace", base.BaseGeometry]):  # type: ignore
        ...

    @lazy_evaluation()
    def intersects(  # type: ignore
        self,
        /,
        shape: Union["Airspace", base.BaseGeometry],
    ) -> bool: ...

    @lazy_evaluation()
    def simplify(  # type: ignore
        self,
        /,
        tolerance: float,
        altitude: Optional[str] = None,
        z_factor: float = 3.048,
    ): ...

    @lazy_evaluation()
    def query_opensky(self):  # type: ignore
        ...

    @lazy_evaluation()
    def query_ehs(self, /, data, failure_mode, propressbar):  # type: ignore
        ...

    @lazy_evaluation()
    def first(self, /, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def last(self, /, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def feature_gt(  # type: ignore
        self,
        /,
        feature: Union[str, Callable[["Flight"], Any]],
        value: Any,
        strict: bool = True,
    ): ...

    @lazy_evaluation()
    def feature_lt(  # type: ignore
        self,
        /,
        feature: Union[str, Callable[["Flight"], Any]],
        value: Any,
        strict: bool = True,
    ): ...

    @lazy_evaluation()
    def shorter_than(  # type: ignore
        self, /, value: Union[str, timedelta, pd.Timedelta], strict: bool = True
    ): ...

    @lazy_evaluation()
    def longer_than(  # type: ignore
        self, /, value: Union[str, timedelta, pd.Timedelta], strict: bool = True
    ): ...

    @lazy_evaluation()
    def max_split(  # type: ignore
        self,
        /,
        value: Union[int, str] = "10T",
        unit: Optional[str] = None,
        key: str = "duration",
    ): ...

    @lazy_evaluation()
    def diff(self, /, features: Union[str, List[str]], **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def apply_segments(  # type: ignore
        self, /, fun: Callable[..., "LazyTraffic"], name: str, *args, **kwargs
    ): ...

    @lazy_evaluation()
    def apply_time(self, /, freq="1T", merge=True, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def agg_time(self, /, freq="1T", merge=True, **kwargs):  # type: ignore
        ...

    @lazy_evaluation()
    def cumulative_distance(  # type: ignore
        self, /, compute_gs: bool = True, compute_track: bool = True, **kwargs
    ): ...

    @lazy_evaluation()
    def compute_wind(self):  # type: ignore
        ...

    @lazy_evaluation()
    def bearing(  # type: ignore
        self,
        other: PointMixin,
        column_name: str = "bearing",
    ): ...

    @lazy_evaluation()
    def distance(  # type: ignore
        self,
        other: Union["Airspace", Polygon, PointMixin],
        column_name: str = "distance",
    ): ...

    @lazy_evaluation()
    def landing_at(self, airport: str) -> bool:  # type: ignore
        ...

    @lazy_evaluation()
    def takeoff_from(self, airport: str) -> bool:  # type: ignore
        ...

    @lazy_evaluation()
    def phases(self, twindow: int = 60):  # type: ignore
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
            return self.query("not onground and altitude.notnull()")
        else:
            return self.query("altitude.notnull()")

    @lazy_evaluation(default=True)
    def onground(self) -> Optional["Traffic"]:
        if "altitude" not in self.data.columns:
            return self
        if "onground" in self.data.columns and self.data.onground.dtype == bool:
            return self.query("onground or altitude.isnull()")
        else:
            return self.query("altitude.isnull()")

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
        if "callsign" not in self.data.columns:
            return set()
        sub = self.data.query("callsign.notnull()")
        if sub.shape[0] == 0:
            return set()
        return set(sub.callsign)

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
        default_key = ["icao24", "callsign"]
        if "callsign" not in self.data.columns:
            default_key = list(elt for elt in default_key if elt != "callsign")
        key = default_key if self.flight_ids is None else "flight_id"
        return (
            self.data.groupby(key)[["timestamp"]]
            .count()
            .sort_values("timestamp", ascending=False)
            .rename(columns={"timestamp": "count"})
        )

    @lazy_evaluation()
    def summary(self, attributes: list[str]) -> pd.DataFrame: ...

    def geoencode(self, *args: Any, **kwargs: Any) -> NoReturn:
        """
        .. danger::

            This method is not implemented.
        """
        raise NotImplementedError

    # -- Visualize with Leaflet --

    def map_leaflet(
        self,
        *,
        zoom: int = 7,
        highlight: Optional[
            Dict[str, Union[str, Flight, Callable[[Flight], Optional[Flight]]]]
        ] = None,
        airport: Union[None, str, Airport] = None,
        **kwargs: Any,
    ) -> Optional[LeafletMap]:
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

    # -- Visualize with Plotly --

    def line_geo(self, **kwargs: Any) -> "go.Figure":
        raise ImportError("Install plotly or traffic with the plotly extension")

    def line_mapbox(
        self, mapbox_style: str = "carto-positron", **kwargs: Any
    ) -> "go.Figure":
        raise ImportError("Install plotly or traffic with the plotly extension")

    def scatter_geo(self, **kwargs: Any) -> "go.Figure":
        raise ImportError("Install plotly or traffic with the plotly extension")

    def scatter_mapbox(
        self, mapbox_style: str = "carto-positron", **kwargs: Any
    ) -> "go.Figure":
        raise ImportError("Install plotly or traffic with the plotly extension")

    def plot(
        self,
        ax: "GeoAxesSubplot",
        nb_flights: Optional[int] = None,
        **kwargs: Any,
    ) -> None:  # coverage: ignore
        """Plots each trajectory on a Matplotlib axis.

        Each Flight supports Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        Example usage:

        >>> from cartes.crs import EuroPP
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
        self, resolution: Union[Dict[str, float], None] = None, **kwargs: Any
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
        ax: "GeoAxesSubplot",
        resolution: Union[Dict[str, float], None] = None,
        threshold: int = 10,
        filtered: bool = False,
        **kwargs: Any,
    ) -> List["Artist"]:  # coverage: ignore
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

        >>> from cartes.crs import Mercator
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

        from cartopy import crs

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

        return ax.barbs(  # type: ignore
            windfield.longitude.to_numpy(),
            windfield.latitude.to_numpy(),
            windfield.wind_u.to_numpy(),
            windfield.wind_v.to_numpy(),
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

        if "callsign" not in self.data.columns:
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
        projection: Union[pyproj.Proj, "crs.Projection", None] = None,
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
        *args: Any,
        projection: Union[None, "crs.Projection", pyproj.Proj] = None,
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

        >>> from cartes.crs import EuroPP
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

    def generation(
        self,
        generation: "GenerationProtocol",
        features: Optional[List[str]] = None,
        scaler: Optional["ScalerProtocol"] = None,
    ) -> Generation:
        """Fits a generative model on the traffic.

        The method:
            - extracts features in underlying Traffic DataFrame to define an
              `X` matrix.
            - *if need be,* apply a scaler to the resulting `X` matrix.
              You may want to consider `StandardScaler()
              <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_;
            - returns a Generation object with a generation model fitted on X.
              You can call sample() on this returned object to generate new
              trajectories.
        """
        if features is None:
            features = ["latitude", "longitude", "altitude", "timedelta"]

        return Generation(generation, features, scaler).fit(self)

    def centroid(
        self,
        nb_samples: Optional[int],
        features: Optional[List[str]] = None,
        projection: Union[None, "crs.Projection", pyproj.Proj] = None,
        transformer: Optional["TransformerProtocol"] = None,
        max_workers: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> "Flight":
        """
        Returns the trajectory in the Traffic that is the closest to all other
        trajectories.

        .. warning::

            Remember the time and space complexity of this method is in O(n^2).

        `*args` and `**kwargs` are passed as is to
        :meth:`scipy.spatial.distance.pdist`

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


def patch_plotly() -> None:
    from ..visualize.plotly import (
        Scattergeo,
        Scattermapbox,
        line_geo,
        line_mapbox,
        scatter_geo,
        scatter_mapbox,
    )

    Traffic.line_mapbox = line_mapbox  # type: ignore
    Traffic.scatter_mapbox = scatter_mapbox  # type: ignore
    Traffic.Scattermapbox = Scattermapbox  # type: ignore
    Traffic.line_geo = line_geo  # type: ignore
    Traffic.scatter_geo = scatter_geo  # type: ignore
    Traffic.Scattergeo = Scattergeo  # type: ignore


try:
    patch_plotly()
except Exception:
    pass


def patch_leaflet() -> None:
    from ..visualize.leaflet import traffic_map_leaflet

    Traffic.map_leaflet = traffic_map_leaflet  # type: ignore


try:
    patch_leaflet()
except Exception:
    pass
