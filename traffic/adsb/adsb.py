from pathlib import Path
from typing import Iterator, Optional

import numpy as np

import pandas as pd


def split(data: pd.DataFrame, value, unit) -> Iterator[pd.DataFrame]:
    diff = data.timestamp.diff().values
    if diff.max() > np.timedelta64(value, unit):
        yield from split(data.iloc[:diff.argmax()], value, unit)
        yield from split(data.iloc[diff.argmax():], value, unit)
    else:
        yield data

class DataFrameWrapper(object):

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def _repr_html_(self):
        return self.data._repr_html_()

class Flight(DataFrameWrapper):

    def plot(self, ax, **kwargs):
        if 'projection' in ax.__dict__:
            from cartopy.crs import PlateCarree
            kwargs['transform'] = PlateCarree()
        ax.plot(self.data.longitude, self.data.latitude, **kwargs)

    def split(self, value=10, unit='m'):
        for data in split(self.data, value, unit):
            yield Flight(data)

    @property
    def start(self):
        return self.data.timestamp.min()

    @property
    def stop(self):
        return self.data.timestamp.max()

class ADSB(DataFrameWrapper):

    def __getitem__(self, index: str) -> Optional[Flight]:
        try:
            value16 = int(index, 16)
            data = self.data[self.data.icao24 == index]
        except ValueError:
            data = self.data[self.data.callsign == index]

        if data.shape[0] > 0:
            return Flight(data)

    def _ipython_key_completions_(self):
        return {*self.data.icao24, *self.data.callsign}

    def __iter__(self):
        for _, df in self.data.groupby('icao24'):
            yield Flight(df)

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

    @classmethod
    def parse_file(self, filename: str) -> 'ADSB':
        path = Path(filename)
        if path.suffixes == ['.pkl']:
            return ADSB(pd.read_pickle(path))
