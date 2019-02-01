# fmt: off

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator,
                    Optional, Set, Tuple, Union, cast)

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

# fmt: on


class Traffic(GeographyMixin):

    _parse_extension: Dict[str, Callable[..., pd.DataFrame]] = dict()

    @classmethod
    def from_flights(cls, flights: Iterable[Optional[Flight]]) -> "Traffic":
        cumul = [f.data for f in flights if f is not None]
        if len(cumul) == 0:
            raise ValueError("empty traffic")
        return cls(pd.concat(cumul, sort=False))

    @classmethod
    def from_file(
        cls, filename: Union[Path, str], **kwargs
    ) -> Optional["Traffic"]:

        tentative = super().from_file(filename)

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

            tentative.data = tentative.data.rename(
                # tentative rename of columns for compatibility
                columns=rename_columns
            )
            return cls(tentative.data)

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

    def __getitem__(self, index: str) -> Optional[Flight]:

        if self.flight_ids is not None:
            data = self.data[self.data.flight_id == index]
            if data.shape[0] > 0:
                return Flight(data)

        # if no such index as flight_id or no flight_id column
        try:
            # If the index can be interpreted as an hexa, it is most likely an
            # icao24 address. However some callsigns may look like an icao24.
            # Tie-breaker:
            #     - if it starts by 0x, priority goes to the icao24;
            #     - if it is in capital letters, priority goes to the callsign
            value16 = int(index, 16)  # noqa: F841 (unused value16)
            if index.startswith("0x"):
                index = index.lower()
                data = self.data.query("icao24 == @index[2:]")
            if index.isupper():
                data = self.data.query("callsign == @index")
            if data.shape[0] == 0:
                index = index.lower()
                data = self.data.query("icao24 == @index")
        except ValueError:
            index = index.upper()
            data = self.data.query("callsign == @index")

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
                yield from Flight(df).split()

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
            Flight(f.data.assign(flight_id=f"{f.callsign}_{id_:>03}"))
            for id_, f in enumerate(self)
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
            params["alpha"] = .8
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

    def resample(self, rule: str = "1s", max_workers: int = 4) -> "Traffic":
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
    ) -> pd.DataFrame:
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

        if projection is None:
            logging.warn("Defaulting to projection EuroPP()")
            projection = crs.EuroPP()

        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

        def yield_pairs(t_chunk: Traffic):
            """
            This function yields all pairs of possible candidates for a CPA
            calculation.
            """

            # combinations types Iterator[Tuple[T, ...]]
            for first, second in cast(
                Iterator[Tuple[Flight, Flight]], combinations(t_chunk, 2)
            ):
                # cast are necessary because of the lru_cache × property bug
                if (
                    cast(pd.Timestamp, first.start)
                    > cast(pd.Timestamp, second.stop)
                ) or (
                    cast(pd.Timestamp, second.start)
                    > cast(pd.Timestamp, first.stop)
                ):
                    # Flights must fly at the same time
                    continue
                if (
                    first.min("altitude")
                    > second.max("altitude") + vertical_separation
                ):
                    # Bounding boxes in altitude must cross
                    continue
                if (
                    second.min("altitude")
                    > first.max("altitude") + vertical_separation
                ):
                    # Bounding boxes in altitude must cross
                    continue
                if first.min("x") > second.max("x") + lateral_separation:
                    # Bounding boxes in x must cross
                    continue
                if second.min("x") > first.max("x") + lateral_separation:
                    # Bounding boxes in x must cross
                    continue
                if first.min("y") > second.max("y") + lateral_separation:
                    # Bounding boxes in y must cross
                    continue
                if second.min("y") > first.max("y") + lateral_separation:
                    # Bounding boxes in y must cross
                    continue

                # Next step is to check the 2D footprint of the trajectories
                # intersect. Before computing the intersection we bufferize the
                # trajectories by half the requested separation.

                first_shape = first.project_shape(projection)
                second_shape = second.project_shape(projection)
                if first_shape is None or second_shape is None:
                    continue

                first_shape = first_shape.simplify(1e3).buffer(
                    lateral_separation / 2
                )
                second_shape = first_shape.simplify(1e3).buffer(
                    lateral_separation / 2
                )

                if first_shape.intersects(second_shape):
                    yield first, second

        t_xyt = (
            self.airborne()
            .compute_xy(projection)
            .assign(round_t=lambda df: df.timestamp.dt.round(round_t))
        )

        cumul = list()

        # Multiprocessing is implemented on each timerange slot only.
        # TODO: it would probably be more efficient to multiprocess over each
        # t_chunk rather than multiprocess the distance computation.

        for _, t_chunk in tqdm(
            t_xyt.groupby("round_t"), total=len(set(t_xyt.data.round_t))
        ):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = {
                    executor.submit(first.distance, second): (
                        first.flight_id,
                        second.flight_id,
                    )
                    for (first, second) in yield_pairs(Traffic(t_chunk))
                }

                for future in as_completed(tasks):
                    cumul.append(future.result())

        return pd.concat(cumul, sort=True)
