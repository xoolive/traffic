from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional, Set

import numpy as np

import pandas as pd
from shapely.geometry import LineString

from ..core.mixins import DataFrameMixin, ShapelyMixin  # type: ignore


def split(data: pd.DataFrame, value, unit) -> Iterator[pd.DataFrame]:
    diff = data.timestamp.diff().values
    if diff.max() > np.timedelta64(value, unit):
        yield from split(data.iloc[:diff.argmax()], value, unit)
        yield from split(data.iloc[diff.argmax():], value, unit)
    else:
        yield data

class Flight(ShapelyMixin, DataFrameMixin):

    def plot(self, ax, **kwargs):
        if 'projection' in ax.__dict__:
            from cartopy.crs import PlateCarree
            kwargs['transform'] = PlateCarree()
        for subflight in self.split(10, 'm'):
            ax.plot(subflight.data.longitude,
                    subflight.data.latitude,
                    **kwargs)

    def split(self, value=10, unit='m'):
        for data in split(self.data, value, unit):
            yield Flight(data)

    @property
    def start(self):
        return self.data.timestamp.min()

    @property
    def stop(self):
        return self.data.timestamp.max()

    @property
    def callsign(self):
        return set(self.data.callsign)

    @property
    def icao24(self):
        return set(self.data.icao24)

    @property
    def shape(self):
        data = self.data[self.data.longitude.notnull()]
        return LineString(zip(data.longitude, data.latitude))

    def _repr_html_(self):
        cumul = ''
        for flight in self.split():
            title = f'<b>{", ".join(flight.callsign)}</b>'
            title += f' ({", ".join(flight.icao24)})'
            no_wrap_div = '<div style="white-space: nowrap">{}</div>'
            cumul += title + no_wrap_div.format(flight._repr_svg_())
        return cumul


class ADSB(DataFrameMixin):

    def __getitem__(self, index: str) -> Optional[Flight]:
        try:
            value16 = int(index, 16)
            data = self.data[self.data.icao24 == index]
        except ValueError:
            data = self.data[self.data.callsign == index]

        if data.shape[0] > 0:
            return Flight(data)

    def _ipython_key_completions_(self):
        return {*self.aircraft, *self.callsigns}

    def __iter__(self):
        for _, df in self.data.groupby('icao24'):
            yield Flight(df)

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

    def query(self, query: str) -> 'ADSB':
        return ADSB(self.data.query(query))

    def resample(self, rule='1s', kernel=(10, 'm')):
        cumul = []
        for flight in self:
            for subflight in flight.split(*kernel):
                cumul.append(subflight.data.assign(
                    start=subflight.start, stop=subflight.stop).
                             set_index('timestamp').
                             resample(rule).
                             interpolate().
                             reset_index().
                             fillna(method='pad'))
        return ADSB(pd.concat(cumul))

    def plot(self, ax, **kwargs):
        for flight in self:
            flight.plot(ax, **kwargs)

    @classmethod
    def parse_file(self, filename: str) -> 'ADSB':
        path = Path(filename)
        if path.suffixes == ['.pkl']:
            return ADSB(pd.read_pickle(path))

