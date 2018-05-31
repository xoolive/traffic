import logging
from datetime import datetime
from typing import Iterator, Optional, Set, Union

import numpy as np

import pandas as pd
from shapely.geometry import LineString

from .mixins import DataFrameMixin, ShapelyMixin


def split(data: pd.DataFrame, value, unit) -> Iterator[pd.DataFrame]:
    diff = data.timestamp.diff().values
    if diff.max() > np.timedelta64(value, unit):
        yield from split(data.iloc[:diff.argmax()], value, unit)
        yield from split(data.iloc[diff.argmax():], value, unit)
    else:
        yield data


class Flight(DataFrameMixin, ShapelyMixin):

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data


    def info_html(self) -> str:
        title = f"<b>Flight {self.callsign}</b>"
        if self.number is not None:
            title += f" / {self.number}"
        if self.flight_id is not None:
            title += f" ({self.flight_id})"

        title += "<ul>"
        title += f"<li><b>aircraft:</b> {self.aircraft}</li>"
        if self.origin is not None:
            title += f"<li><b>origin:</b> {self.origin} ({self.start})</li>"
        else:
            title += f"<li><b>origin:</b> {self.start}</li>"
        if self.destination is not None:
            title += f"<li><b>destination:</b> {self.destination} "
            title +=f"({self.stop})</li>"
        else:
            title += f"<li><b>destination:</b> {self.stop}</li>"
        title += "</ul>"
        return title

    def _repr_html_(self):
        title = self.info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())


    @property
    def start(self) -> datetime:
        return self.data.timestamp.min()

    @property
    def stop(self) -> datetime:
        return self.data.timestamp.max()

    @property
    def callsign(self) -> Union[str, Set[str]]:
        tmp = set(self.data.callsign)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several callsigns for one flight, consider splitting")
        return tmp

    @property
    def number(self) -> Optional[Union[str, Set[str]]]:
        if 'number' not in self.data.columns:
            return None
        tmp = set(self.data.number)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several numbers for one flight, consider splitting")
        return tmp

    @property
    def icao24(self) -> Union[str, Set[str]]:
        tmp = set(self.data.icao24)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several icao24 for one flight, consider splitting")
        return tmp

    @property
    def flight_id(self) -> Optional[Union[str, Set[str]]]:
        if 'flight_id' not in self.data.columns:
            return None
        tmp = set(self.data.flight_id)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several ids for one flight, consider splitting")
        return tmp

    @property
    def origin(self) -> Optional[Union[str, Set[str]]]:
        if 'origin' not in self.data.columns:
            return None
        tmp = set(self.data.origin)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several origins for one flight, consider splitting")
        return tmp

    @property
    def destination(self) -> Optional[Union[str, Set[str]]]:
        if 'destination' not in self.data.columns:
            return None
        tmp = set(self.data.destination)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several destinations for one flight, consider splitting")
        return tmp

    @property
    def aircraft(self) -> Optional[str]:
        if not isinstance(self.icao24, str):
            return None
        from ..data import aircraft as acdb
        ac = acdb[self.icao24]
        if ac.shape[0] != 1:
            return self.icao24
        else:
            return f"{self.icao24} / {ac.iloc[0].regid} ({ac.iloc[0].mdl})"

    @property
    def linestring(self) -> LineString:
        data = self.data[self.data.longitude.notnull()]
        return LineString(zip(data['longitude'], data['latitude']))

    @property
    def shape(self) -> LineString:
        return self.linestring

    def plot(self, ax, **kwargs):
        if 'projection' in ax.__dict__ and 'transform' not in kwargs:
            from cartopy.crs import PlateCarree
            kwargs['transform'] = PlateCarree()

        if 'color' not in kwargs:
            kwargs['color'] = '#aaaaaa'

        ax.plot(*self.shape.xy, **kwargs)

    def split(self, value: int=10, unit: str='m') -> Iterator['Flight']:
        for data in split(self.data, value, unit):
            yield Flight(data)

    def resample(self, rule: str='1s') -> 'Flight':
        data = (self.data.assign(start=self.start, stop=self.stop).
                set_index('timestamp').
                resample(rule).
                interpolate().
                reset_index().
                fillna(method='pad'))
        return Flight(data)

