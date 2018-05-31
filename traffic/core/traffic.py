from functools import lru_cache
from typing import Iterator, Optional, Set

import pandas as pd

from .flight import Flight
from .mixins import DataFrameMixin


class Traffic(DataFrameMixin):

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

    def __iter__(self):
        if self.flight_ids is not None:
            for _, df in self.data.groupby('flight_id'):
                yield Flight(df)
        else:
            for _, df in self.data.groupby('icao24'):
                yield from Flight(df).split()


    @property
    @lru_cache()
    def callsigns(self) -> Set[str]:
        """Return only the most relevant callsigns"""
        sub = (self.data.query('callsign == callsign').
               groupby(('callsign', 'icao24')).
               filter(lambda x: len(x) > 10))
        return set(cs for cs in sub.callsign if len(cs) > 3 and " " not in cs)

    @property
    def aircraft(self) -> Set[str]:
        return set(self.data.icao24)

    @property
    def flight_ids(self) -> Optional[Set[str]]:
        if 'flight_id' in self.data.columns:
            return set(self.data.flight_id)
        return None

    def resample(self, rule='1s', kernel=(10, 'm')):
        cumul = []
        for flight in self:
            for subflight in flight.split(*kernel):
                cumul.append(subflight.resample(rule))
        return Traffic.fromFlights(cumul)

    def plot(self, ax, **kwargs):
        for flight in self:
            flight.plot(ax, **kwargs)

    @classmethod
    def from_flights(cls, flights: Iterator[Flight]):
        return cls(pd.concat([f.data for f in flights]))
