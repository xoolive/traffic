from calendar import timegm
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple, Union

import numpy as np

import maya
import pandas as pd
from cartopy.crs import PlateCarree
from fastkml import kml
from fastkml.geometry import Geometry
from scipy.interpolate import interp1d
from shapely.geometry import LineString
from shapely.geometry.base import BaseMultipartGeometry

from ..data.airac import Sector
from ..kml import toStyle

timelike = Union[str, int, datetime]
time_or_delta = Union[timelike, timedelta]


def time(int_: int) -> datetime:
    ts = timegm((2000 + int_ // 10000,
                 int_ // 100 % 100,
                 int_ % 100, 0, 0, 0))
    return datetime.fromtimestamp(ts)


def hour(int_: int) -> timedelta:
    return timedelta(hours=int_ // 10000,
                     minutes=int_ // 100 % 100,
                     seconds=int_ % 100)


def to_datetime(time: timelike) -> datetime:
    if isinstance(time, str):
        time = maya.parse(time).epoch
    if isinstance(time, int):
        time = datetime.fromtimestamp(time)
    return time


class Flight(object):

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.interpolator: Dict = dict()

    def _repr_svg_(self):
        print(f"{self.callsign}: {self.origin} ({self.start})"
              f" → {self.destination} ({self.stop})")
        return self.linestring._repr_svg_()

    def __len__(self) -> int:
        return self.data.shape[0]

    def plot(self, ax, **kwargs):
        if 'projection' in ax.__dict__ and 'transform' not in kwargs:
            kwargs['transform'] = PlateCarree()

        if 'color' not in kwargs:
            kwargs['color'] = '#aaaaaa'

        for _, segment in self.data.iterrows():
            ax.plot([segment.lon1, segment.lon2],
                    [segment.lat1, segment.lat2], **kwargs)

    @property
    def callsign(self) -> int:
        return self.data.iloc[0].callsign

    @property
    def start(self) -> datetime:
        return min(self.times)

    @property
    def stop(self) -> datetime:
        return max(self.times)

    @property
    def origin(self) -> str:
        return self.data.iloc[0].origin

    @property
    def destination(self) -> str:
        return self.data.iloc[0].destination

    @property
    def aircraft(self) -> str:
        return self.data.iloc[0].aircraft

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.lon1, s.lat1, s.alt1
        yield s.lon2, s.lat2, s.alt2

    @property
    def times(self) -> Iterator[datetime]:
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.time1
        yield s.time2

    @property
    def linestring(self) -> LineString:
        return LineString(list(self.coords))

    def interpolate(self, times, proj=PlateCarree()):
        """Interpolates a trajectory in time.  """
        if proj not in self.interpolator:
            self.interpolator[proj] = interp1d(
                np.stack(t.timestamp() for t in self.times),
                proj.transform_points(PlateCarree(),
                                      *np.stack(self.coords).T).T)
        return PlateCarree().transform_points(
            proj, *self.interpolator[proj](times))

    def at(self, time: timelike, proj=PlateCarree()) -> np.ndarray:
        time = to_datetime(time)
        timearray: np.ndarray[datetime] = np.array([time.timestamp()])
        return self.interpolate(timearray, proj)

    def between(self, before: timelike, after: time_or_delta) -> 'Flight':
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)

        t: np.ndarray = np.stack(self.times)
        index = np.where((before < t) & (t < after))

        new_data: np.ndarray = np.vstack([self.at(before),
                                          np.stack(self.coords)[index],
                                          self.at(after)])

        df: pd.DataFrame = (
            pd.DataFrame.from_records(
                np.c_[new_data[:-1, :], new_data[1:, :]],
                columns=['lon1', 'lat1', 'alt1', 'lon2', 'lat2', 'alt2']).
            assign(time1=[before, *t[index]],
                   time2=[*t[index], after],
                   origin=self.origin,
                   destination=self.destination,
                   aircraft=self.aircraft,
                   callsign=self.callsign))

        return Flight(df)

    def intersects(self, sector: Sector):
        for layer in sector:
            ix = self.linestring.intersection(layer.polygon)
            if not ix.is_empty:
                if isinstance(ix, BaseMultipartGeometry):
                    # TODO this sounds plausible yet weird...
                    for part in ix:
                        if any(100*layer.lower < x[2] < 100*layer.upper
                               for x in part.coords):
                            return True
                else:
                    if any(layer.lower < x[2] < layer.upper
                           for x in ix.coords):
                        return True
        return False

    def export_kml(self, styleUrl:Optional[kml.StyleUrl]=None,
                   color:Optional[str]=None, alpha:float=.5, **kwargs):
        if color is not None:
            styleUrl = toStyle(color)
        params = {'name': self.callsign,
                  'description': f"{self.origin} → {self.destination}",
                  'styleUrl': styleUrl}
        for key, value in kwargs.items():
            params[key] = value
        placemark = kml.Placemark(**params)
        placemark.visibility = 1
        # Convert to meters
        coords = np.stack(self.coords)
        coords[:, 2] *= 0.3048
        placemark.geometry = Geometry(geometry=LineString(coords),
                                      extrude=True,
                                      altitude_mode='relativeToGround')
        return placemark


class SO6(object):

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    def __getitem__(self, callsign: str) -> Flight:
        return Flight(self.data.groupby('callsign').get_group(callsign))

    def __iter__(self) -> Iterator[Tuple[str, Flight]]:
        for callsign, flight in self.data.groupby('callsign'):
            yield callsign, Flight(flight)

    def __len__(self) -> int:
        return len(self.callsigns)

    def _ipython_key_completions_(self):
        return self.callsigns

    @property
    def callsigns(self) -> Set[str]:
        return set(self.data.callsign)

    @classmethod
    def parse_so6(self, filename: str) -> 'SO6':
        so6 = pd.read_csv(filename, sep=" ", header=-1,
                          names=['d1', 'origin', 'destination', 'aircraft',
                                 'hour1', 'hour2', 'alt1', 'alt2', 'd2',
                                 'callsign', 'date1', 'date2',
                                 'lat1', 'lon1', 'lat2', 'lon2',
                                 'd3', 'd4', 'd5', 'd6'])

        so6 = so6.assign(lat1=so6.lat1/60, lat2=so6.lat2/60,
                         lon1=so6.lon1/60, lon2=so6.lon2/60,
                         alt1=so6.alt1*100, alt2=so6.alt2*100,
                         time1=so6.date1.apply(time) + so6.hour1.apply(hour),
                         time2=so6.date2.apply(time) + so6.hour2.apply(hour),)

        for col in ('d1', 'd2', 'd3', 'd4', 'd5', 'd6',
                    'date1', 'date2', 'hour1', 'hour2'):
            del so6[col]

        return SO6(so6)

    @classmethod
    def parse_pkl(self, filename: str) -> 'SO6':
        so6 = pd.read_pickle(filename)
        return SO6(so6)

    def to_pkl(self, filename: str) -> None:
        self.data.to_pickle(filename)

    def at(self, time: timelike) -> 'SO6':
        time = to_datetime(time)
        return SO6(self.data[(self.data.time1 <= time) &
                             (self.data.time2 > time)])

    def between(self, before: timelike, after: time_or_delta) -> 'SO6':
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)
        return SO6(self.data[(self.data.time1 <= after) &
                             (self.data.time2 >= before)])

    def intersects(self, sector) -> 'SO6':
        return SO6(self.data.
                   groupby('callsign').
                   filter(lambda flight: Flight(flight).intersects(sector)))

    def inside_bbox(self, bounds: Union[Sector, Tuple[float, ...]]) -> 'SO6':

        if isinstance(bounds, Sector):
           bounds = bounds.flatten().bounds

        west, south, east, north = bounds

        # Transform coords into intelligible floats
        # '-2.06 <= lon1 <= 4.50 & 42.36 <= lat1 <= 48.14', instead of
        #  (-2.066666603088379, 42.366943359375, 4.491666793823242,
        #   48.13333511352539)

        dec = Decimal('0.00')
        west = Decimal(west).quantize(dec, rounding=ROUND_DOWN)
        east = Decimal(east).quantize(dec, rounding=ROUND_UP)
        south = Decimal(south).quantize(dec, rounding=ROUND_DOWN)
        north = Decimal(north).quantize(dec, rounding=ROUND_UP)

        # the numexpr query is 10% faster than the regular
        # data[data.lat1 >= ...] conjunctions of comparisons
        query = "{0} <= lon1 <= {2} & {1} <= lat1 <= {3}"
        query = query.format(west, south, east, north)

        data = self.data.query(query)

        callsigns: Set[str] = set(data.callsign)

        return SO6(self.data.groupby('callsign').filter(
            lambda data: data.iloc[0].callsign in callsigns))

    def select(self, query: Union['SO6', Iterable[str]]) -> 'SO6':
        if isinstance(query, SO6):
            query = query.callsigns
        select = self.data.callsign.isin(query)
        return SO6(self.data[select])
