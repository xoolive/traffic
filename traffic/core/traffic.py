# fmt: off

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator,
                    List, Optional, Set, Tuple, Type, TypeVar, Union, overload)

import pandas as pd
import pyproj
from cartopy import crs
from cartopy.mpl.geoaxes import GeoAxesSubplot
from tqdm.autonotebook import tqdm

from ..core.time import time_or_delta, timelike, to_datetime
from .flight import Flight
from .mixins import GeographyMixin
from .sv import StateVectors

if TYPE_CHECKING:
    from .airspace import Airspace  # noqa: F401
    from ..algorithms.cpa import CPA  # noqa: F401

# fmt: on

# https://github.com/python/mypy/issues/2511
TrafficTypeVar = TypeVar("TrafficTypeVar", bound="Traffic")


class Traffic(GeographyMixin):

    __slots__ = ("data",)

    _parse_extension: Dict[str, Callable[..., pd.DataFrame]] = dict()

    @classmethod
    def from_flights(cls, flights: Iterable[Optional[Flight]]) -> "Traffic":
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

    # TODO Iterable[str] would be more appropriate but it overlaps with str
    @overload  # noqa: F811
    def __getitem__(
        self, index: Union[List[str], Set[str]]
    ) -> Optional["Traffic"]:
        ...

    def __getitem__(self, index):  # noqa: F811

        data = self.data  # should be useless except in some cornercase

        if not isinstance(index, str):
            logging.debug("Selecting flights from a list of identifiers")
            subset = list(index)  # noqa: F841
            if "flight_id" in self.data.columns:
                return self.__class__(
                    self.data.loc[self.data.flight_id.isin(subset)]
                )
            else:
                return self.__class__(
                    self.data.loc[self.data.callsign.isin(subset)]
                )

        if self.flight_ids is not None:
            data = data[data.flight_id == index]
            if data.shape[0] > 0:
                return Flight(data)

        logging.debug("Fallbacking to icao24/callsign")

        # if no such index as flight_id or no flight_id column
        try:
            # If the index can be interpreted as an hexa, it is most likely an
            # icao24 address. However some callsigns may look like an icao24.
            # Tie-breaker:
            #     - if it starts by 0x, priority goes to the icao24;
            #     - if it is in capital letters, priority goes to the callsign
            value16 = int(index, 16)  # noqa: F841 (unused value16)
            default_icao24 = True
            if index.startswith("0x"):
                index = index.lower()
                logging.debug("Selecting an icao24")
                data = self.data.loc[self.data.icao24 == index[2:]]
                default_icao24 = False
            if index.isupper():
                logging.debug("Selecting a callsign")
                data = self.data.loc[self.data.callsign == index]
                if data.shape[0] > 0:
                    default_icao24 = False
            if default_icao24:
                index = index.lower()
                logging.debug("Selecting an icao24")
                data = self.data.loc[self.data.icao24 == index]
        except ValueError:
            index = index.upper()
            logging.debug("Selecting a callsign")
            data = self.data.loc[self.data.callsign == index]

        if data.shape[0] > 0:
            return Flight(data)

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

    def filter_if(self, criterion: Callable[[Flight], bool]) -> "Traffic":
        return Traffic.from_flights(
            flight for flight in self if criterion(flight)
        )

    def subset(self, callsigns: Iterable[str]) -> "Traffic":
        warnings.warn("Use filter_if instead", DeprecationWarning)
        if "flight_id" in self.data.columns:
            return Traffic.from_flights(
                flight
                for flight in self
                # should not be necessary but for type consistency
                if flight.flight_id is not None
                and flight.flight_id in callsigns
            )
        else:
            return Traffic.from_flights(
                flight
                for flight in self
                if flight.callsign in callsigns  # type: ignore
            )

    # --- Properties ---

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def start_time(self) -> pd.Timestamp:
        return self.data.timestamp.min()

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def end_time(self) -> pd.Timestamp:
        return self.data.timestamp.max()

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache()
    def callsigns(self) -> Set[str]:
        """Return only the most relevant callsigns"""
        sub = self.data.query("callsign == callsign")
        return set(cs for cs in sub.callsign if len(cs) > 3 and " " not in cs)

    @property
    def aircraft(self) -> Set[str]:
        return set(self.data.icao24)

    @property
    def flight_ids(self) -> Optional[Set[str]]:
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
                if flight.start <= time <= flight.stop  # type: ignore
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

    def before(self, ts: timelike) -> "Traffic":
        return self.between(self.start_time, ts)

    def after(self, ts: timelike) -> "Traffic":
        return self.between(ts, self.end_time)

    def between(self, before: timelike, after: time_or_delta) -> "Traffic":
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)

        # full call is necessary to keep @before and @after as local variables
        # return self.query('@before < timestamp < @after')  => not valid
        return self.__class__(self.data.query("@before < timestamp < @after"))

    def airborne(self) -> "Traffic":
        """Returns the airborne part of the Traffic.

        The airborne part is determined by null values on the altitude column.
        """
        return self.query("altitude == altitude")

    @lru_cache()
    def stats(self) -> pd.DataFrame:
        """Statistics about flights contained in the structure.
        Useful for a meaningful representation.
        """
        key = ["icao24", "callsign"] if self.flight_ids is None else "flight_id"
        return (
            self.data.groupby(key)[["timestamp"]]
            .count()
            .sort_values("timestamp", ascending=False)
            .rename(columns={"timestamp": "count"})
        )

    def assign_id(self) -> "Traffic":
        if "flight_id" in self.data.columns:
            return self
        return Traffic.from_flights(
            flight.assign(flight_id=f"{flight.callsign}_{id_:>03}")
            for id_, flight in enumerate(self)
        )

    def filter(
        self,
        strategy: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = lambda x: x.bfill().ffill(),
        **kwargs,
    ) -> "Traffic":
        return Traffic.from_flights(
            flight.filter(strategy, **kwargs) for flight in self
        )

    def plot(
        self, ax: GeoAxesSubplot, nb_flights: Optional[int] = None, **kwargs
    ) -> None:
        params: Dict[str, Any] = {}
        if sum(1 for _ in zip(range(8), self)) == 8:
            params["color"] = "#aaaaaa"
            params["linewidth"] = 1
            params["alpha"] = 0.8
            kwargs = {**params, **kwargs}  # precedence of kwargs over params
        for i, flight in enumerate(self):
            if nb_flights is None or i < nb_flights:
                flight.plot(ax, **kwargs)

    @property
    def widget(self):
        from ..drawing.ipywidgets import TrafficWidget

        return TrafficWidget(self)

    def inside_bbox(
        self, bounds: Union["Airspace", Tuple[float, ...]]
    ) -> "Traffic":
        # implemented and monkey-patched in airspace.py
        # given here for consistency in types
        raise NotImplementedError

    def intersects(self, airspace: "Airspace") -> "Traffic":
        # implemented and monkey-patched in airspace.py
        # given here for consistency in types
        raise NotImplementedError

    # --- Real work ---

    def resample(
        self, rule: Union[str, int] = "1s", max_workers: int = 4
    ) -> "Traffic":
        """Resamples all trajectories, flight by flight.

        `rule` defines the desired sample rate (default: 1s)
        """

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            cumul = []
            tasks = {
                executor.submit(flight.resample, rule): flight
                for flight in self
            }
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                cumul.append(future.result())

        return self.__class__.from_flights(cumul)

    def closest_point_of_approach(
        self,
        lateral_separation: float,
        vertical_separation: float,
        projection: Union[pyproj.Proj, crs.Projection, None] = None,
        round_t: str = "d",
        max_workers: int = 4,
    ) -> "CPA":
        """
        Computes a CPA dataframe for all pairs of trajectories candidates for
        being separated by less than lateral_separation in vertical_separation.

        In order to be computed efficiently, the method needs the following
        parameters:

        - projection: a first filtering is applied on the bounding boxes of
        trajectories, expressed in meters. You need to provide a decent
        projection able to approximate distances by Euclide formula.
        By default, EuroPP() projection is considered, but a non explicit
        argument will raise a warning.

        - round_t: an additional column will be added in the DataFrame to group
        trajectories by relevant time frames. Distance computations will be
        considered only between trajectories flown in the same time frame.
        By default, the 'd' pandas freq parameter is considered, to group
        trajectories by day, but other ways of splitting ('h') may be more
        relevant and impact performance.

        - max_workers: distance computations are spread over a given number of
        processors.

        """

        from ..algorithms.cpa import closest_point_of_approach

        return closest_point_of_approach(
            self,
            lateral_separation,
            vertical_separation,
            projection,
            round_t,
            max_workers,
        )
