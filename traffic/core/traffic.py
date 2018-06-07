from functools import lru_cache
from typing import Iterator, Optional, Set

import pandas as pd
import pyproj

from .flight import Flight
from .mixins import DataFrameMixin


class Traffic(DataFrameMixin):

    @classmethod
    def from_flights(cls, flights: Iterator[Flight]):
        return cls(pd.concat([f.data for f in flights]))

    # --- Special methods ---

    def __add__(self, other) -> 'Traffic':
        # useful for compatibility with sum() function
        if other == 0: return self
        return Traffic(pd.concat([self.data, other.data]))

    def __radd__(self, other) -> 'Traffic':
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

    def __iter__(self):
        if self.flight_ids is not None:
            for _, df in self.data.groupby('flight_id'):
                yield Flight(df)
        else:
            for _, df in self.data.groupby('icao24'):
                yield from Flight(df).split()

    def __repr__(self):
        return self.stats().__repr__()

    def _repr_html_(self):
        stats = self.stats()
        shape = stats.shape[0]
        if shape > 10:
            # stylers are not efficient on big dataframes...
            stats = stats.head(10)
        styler = stats.style.bar(align='mid', color='#5fba7d')
        rep = f"<b>Traffic with {shape} identifiers</b>"
        return rep + styler._repr_html_()

    # --- Properties ---

    @property
    def start_time(self):
        return self.data.timestamp.min()

    @property
    def end_time(self):
        return self.data.timestamp.max()

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
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

    # --- Easy work ---

    @lru_cache()
    def stats(self) -> pd.DataFrame:
        """Statistics about flights contained in the structure.
        Useful for a meaningful representation.
        """
        key = ('icao24', 'callsign') if self.flight_ids is None else 'flight_id'
        return (self.data.groupby(key)[['timestamp']].count().
                sort_values('timestamp', ascending=False).
                rename(columns={'timestamp': 'count'}))

    def plot(self, ax, **kwargs):
        for flight in self:
            flight.plot(ax, **kwargs)

    # --- Real work ---

    def resample(self, rule='1s', kernel=(10, 'm')):
        """Resamples all trajectories, flight by flight.

        `rule` defines the desired sample rate (default: 1s)
        `kernel` defines how to iter on flights (see `Flight.split`)
        """
        cumul = []
        for flight in self:
            for subflight in flight.split(*kernel):
                cumul.append(subflight.resample(rule))
        return Traffic.from_flights(cumul)

    def compute_xy(self, projection=None) -> 'Traffic':
        """Computes x and y columns from latitudes and longitudes.

        The source projection is WGS84 (EPSG 4326).
        The default destination projection is a Lambert Conformal Conical
        projection centered on the data inside the dataframe.

        For consistency reasons with pandas DataFrame, a new Traffic structure
        is returned.

        """
        if projection is None:
            projection = pyproj.Proj(proj='lcc',
                                     lat_1=self.data.latitude.min(),
                                     lat_2=self.data.latitude.max(),
                                     lat_0=self.data.latitude.mean(),
                                     lon_0=self.data.longitude.mean())

        x, y = pyproj.transform(pyproj.Proj(init='EPSG:4326'),
                                projection,
                                self.data.longitude.values,
                                self.data.latitude.values)

        return Traffic(self.data.assign(x=x, y=y))
