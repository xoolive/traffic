# fmt: off

import logging
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator,
                    List, Optional, Set, Type, TypeVar, Union, overload)

import pandas as pd
import pyproj
from cartopy import crs
from cartopy.mpl.geoaxes import GeoAxesSubplot
from shapely.geometry import base

from ..algorithms.clustering import Clustering, centroid
from ..algorithms.cpa import closest_point_of_approach
from ..core.time import time_or_delta, timelike, to_datetime
from .flight import Flight
from .lazy import lazy_evaluation
from .mixins import GeographyMixin
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


class Traffic(GeographyMixin):
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
    _parse_extension: Dict[str, Callable[..., pd.DataFrame]] = dict()

    @classmethod
    def from_flights(cls, flights: Iterable[Optional[Flight]]) -> "Traffic":
        """
        Creates a Traffic structure from all flights passed as an
        iterator or iterable.
        """
        cumul = [f.data for f in flights if f is not None]
        if len(cumul) == 0:
            raise ValueError("empty traffic")
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
            }

            if (
                "baroaltitude" in tentative.data.columns
                or "baro_altitude" in tentative.data.columns
            ):
                # for retrocompatibility
                rename_columns["altitude"] = "geoaltitude"

            return tentative.rename(columns=rename_columns)

        path = Path(filename)
        method = cls._parse_extension.get("".join(path.suffixes), None)
        if method is None:
            logging.warn(f"{path.suffixes} extension is not supported")
            return None

        data = method(filename, **kwargs)
        if data is None:
            return None

        return cls(data)

    # --- Special methods ---

    def __add__(self, other) -> "Traffic":
        # useful for compatibility with sum() function
        if other == 0:
            return self
        return self.__class__(pd.concat([self.data, other.data], sort=False))

    def __radd__(self, other) -> "Traffic":
        return self + other

    @overload
    def __getitem__(self, index: str) -> Optional[Flight]:
        ...

    @overload  # noqa: F811
    def __getitem__(self, index: IterStr) -> "Traffic":
        ...

    def __getitem__(self, index):  # noqa: F811

        if not isinstance(index, str):
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

    def __iter__(self) -> Iterator[Flight]:
        if self.flight_ids is not None:
            for _, df in self.data.groupby("flight_id"):
                yield Flight(df)
        else:
            for _, df in self.data.groupby(["icao24", "callsign"]):
                yield from Flight(df).split("10 minutes")

    def __len__(self):
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        stats = self.stats()
        shape = stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            stats = stats.head(10)
        return stats.__repr__()

    def _repr_html_(self) -> str:
        stats = self.stats()
        shape = stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            stats = stats.head(10)
        styler = stats.style.bar(align="mid", color="#5fba7d")
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
    def unwrap(self, features: Union[str, List[str]] = ["track", "heading"]):
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

    # -- Methods with a Traffic implementation, otherwise delegated to Flight

    @lazy_evaluation(default=True)
    def before(self, ts: timelike) -> "Traffic":
        return self.between(self.start_time, ts)

    @lazy_evaluation(default=True)
    def after(self, ts: timelike) -> "Traffic":
        return self.between(ts, self.end_time)

    @lazy_evaluation(default=True)
    def between(self, before: timelike, after: time_or_delta) -> "Traffic":

        before = to_datetime(before)

        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)

        # full call is necessary to keep @before and @after as local variables
        # return self.query('@before < timestamp < @after')  => not valid
        return self.__class__(self.data.query("@before < timestamp < @after"))

    @lazy_evaluation(default=True)
    def airborne(self) -> "Traffic":
        """Returns the airborne part of the Traffic.

        The airborne part is determined by an ``onground`` flag or null values
        in the altitude column.
        """
        if "onground" in self.data.columns and self.data.onground.dtype == bool:
            return self.query("not onground and altitude == altitude")
        else:
            return self.query("altitude == altitude")

    # --- Properties ---

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def start_time(self) -> pd.Timestamp:
        """Returns the earliest timestamp in the DataFrame."""
        return self.data.timestamp.min()

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def end_time(self) -> pd.Timestamp:
        """Returns the latest timestamp in the DataFrame."""
        return self.data.timestamp.max()

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def callsigns(self) -> Set[str]:
        """Return all the different callsigns in the DataFrame"""
        sub = self.data.query("callsign == callsign")
        return set(sub.callsign)

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def aircraft(self) -> Set[str]:
        """Return all the different icao24 aircraft ids in the DataFrame"""
        return set(self.data.icao24)

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def flight_ids(self) -> Optional[Set[str]]:
        """Return all the different flight_id in the DataFrame"""
        if "flight_id" in self.data.columns:
            return set(self.data.flight_id)
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

    @lru_cache()
    def stats(self) -> pd.DataFrame:
        key = ["icao24", "callsign"] if self.flight_ids is None else "flight_id"
        return (
            self.data.groupby(key)[["timestamp"]]
            .count()
            .sort_values("timestamp", ascending=False)
            .rename(columns={"timestamp": "count"})
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
        >>> fig, ax = plt.subplots(1, subplot_kw=dict(projection=EuroPP())
        >>> t.plot(ax, alpha=.5)

        """
        params: Dict[str, Any] = {}
        if sum(1 for _ in zip(range(8), self)) == 8:
            params["color"] = "#aaaaaa"
            params["linewidth"] = 1
            params["alpha"] = 0.8
            kwargs = {**params, **kwargs}  # precedence of kwargs over params
        for i, flight in enumerate(self):
            if nb_flights is None or i < nb_flights:
                flight.plot(ax, **kwargs)

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
    ) -> "CPA":
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
        nb_samples: int,
        features: List[str] = ["x", "y"],
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
        ...     method=DBSCAN(eps=1.5, min_samples=10),
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
        nb_samples: int,
        features: List[str] = ["x", "y"],
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
