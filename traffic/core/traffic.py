# fmt: off

import logging
from functools import lru_cache
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, Optional, Set,
                    Union)

import pandas as pd
from cartopy.mpl.geoaxes import GeoAxesSubplot

from ..core.time import time_or_delta, timelike, to_datetime
from .flight import Flight
from .mixins import DataFrameMixin, GeographyMixin

# fmt: on


class Traffic(DataFrameMixin, GeographyMixin):

    _parse_extension: Dict[str, Callable[..., pd.DataFrame]] = dict()

    @classmethod
    def from_flights(cls, flights: Iterable[Flight]):
        cumul = [f.data for f in flights if f is not None]
        if len(cumul) == 0:
            return None
        return cls(pd.concat(cumul, sort=False))

    @classmethod
    def from_file(
        cls, filename: Union[Path, str], **kwargs
    ) -> Optional["Traffic"]:

        tentative = super().from_file(filename)
        if tentative is not None:
            tentative.data = tentative.data.rename(
                # tentative rename of columns for compatibility
                columns={
                    "lat": "latitude",
                    "lon": "longitude",
                    "velocity": "ground_speed",
                    "groundspeed": "ground_speed",
                    "vertrate": "vertical_rate",
                    "roc": "vertical_rate",
                    "baroaltitude": "baro_altitude",
                    "geoaltitude": "altitude",
                    "time": "timestamp",
                }
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
            value16 = int(index, 16)  # noqa: F841 (unused value16)
            data = self.data[self.data.icao24 == index]
        except ValueError:
            data = self.data[self.data.callsign == index]

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
        return self.stats().__repr__()

    def _repr_html_(self) -> str:
        stats = self.stats()
        shape = stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            stats = stats.head(10)
        styler = stats.style.bar(align="mid", color="#5fba7d")
        rep = f"<b>Traffic with {shape} identifiers</b>"
        return rep + styler._repr_html_()

    def subset(self, callsigns: Iterable[str]) -> "Traffic":
        return Traffic.from_flights(f for f in self if f.callsign in callsigns)

    # --- Properties ---

    @property
    def start_time(self) -> pd.Timestamp:
        return self.data.timestamp.min()

    @property
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

    def at(self, time: timelike) -> "Traffic":
        time = to_datetime(time)
        list_flights = [
            flight.at(time)
            for flight in self
            if flight.start <= time <= flight.stop
        ]
        return Traffic(
            pd.DataFrame.from_records(
                [s for s in list_flights if s is not None]
            ).assign(
                # attribute 'name' refers to the index, i.e. 'timestamp'
                timestamp=[s.name for s in list_flights if s is not None]
            )
        )

    def before(self, ts: timelike) -> "Traffic":
        return Traffic.from_flights(flight.before(ts) for flight in self)

    def after(self, ts: timelike) -> "Traffic":
        return Traffic.from_flights(flight.after(ts) for flight in self)

    def between(self, before: timelike, after: time_or_delta) -> "Traffic":
        return Traffic.from_flights(
            flight.between(before, after) for flight in self
        )

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

    # --- Real work ---

    def resample(self, rule="1s", kernel=(10, "m")):
        """Resamples all trajectories, flight by flight.

        `rule` defines the desired sample rate (default: 1s)
        `kernel` defines how to iter on flights (see `Flight.split`)
        """
        cumul = []
        for flight in self:
            for subflight in flight.split(*kernel):
                cumul.append(subflight.resample(rule))
        return self.__class__.from_flights(cumul)
