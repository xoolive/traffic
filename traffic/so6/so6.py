from calendar import timegm
from datetime import datetime, timedelta

import maya
import numpy as np
import pandas as pd

from shapely.geometry import LineString
from scipy.interpolate import interp1d
from cartopy.crs import PlateCarree

def time(int_):
    ts = timegm((2000 + int_//10000, int_//100 % 100, int_ % 100, 0, 0, 0))
    return datetime.fromtimestamp(ts)

def hour(int_):
    return timedelta(hours=int_//10000,
                     minutes=int_//100 % 100,
                     seconds=int_ % 100)

def to_datetime(time):
    if isinstance(time, str):
        time = maya.parse(time).epoch
    if isinstance(time, int):
        time = datetime.fromtimestamp(time)
    return time


class Flight(object):

    def __init__(self, data):
        self.data = data
        self.interpolator = dict()

    def _repr_svg_(self):
        return self.linestring._repr_svg_()

    def plot(self, ax, **kwargs):
        if 'projection' in ax.__dict__ and 'transform' not in kwargs:
            kwargs['transform'] = PlateCarree()

        if 'color' not in kwargs:
            kwargs['color'] = '#aaaaaa'

        for _, segment in self.data.iterrows():
            ax.plot([segment.lon1, segment.lon2],
                    [segment.lat1, segment.lat2], **kwargs)

    @property
    def callsign(self):
        return self.data.iloc[0].callsign

    @property
    def start(self):
        return min(self.times())

    @property
    def stop(self):
        return max(self.times())

    @property
    def origin(self):
        return self.data.iloc[0].origin

    @property
    def destination(self):
        return self.data.iloc[0].destination

    @property
    def aircraft(self):
        return self.data.iloc[0].aircraft

    @property
    def coords(self):
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.lon1, s.lat1, s.alt1
        yield s.lon2, s.lat2, s.alt2

    @property
    def times(self):
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.time1
        yield s.time2

    @property
    def linestring(self):
        return LineString(list(self.coords))

    def interpolate(self, times, proj=PlateCarree()):
        """Interpolates a trajectory in time.

        times:  a numpy array of timestamps
        """
        if proj not in self.interpolator:
            self.interpolator[proj] = interp1d(
                np.stack(t.timestamp() for t in self.times),
                proj.transform_points(PlateCarree(),
                                      *np.stack(self.coords).T).T)
        return PlateCarree().transform_points(
            proj, *self.interpolator[proj](times))

    def at(self, time, proj=PlateCarree()):
        time = to_datetime(time)
        return self.interpolate(np.array([time.timestamp()]), proj)

    def between(self, before, after):
        before, after = to_datetime(before), to_datetime(after)

        t = np.stack(self.times)
        index = np.where((before < t) & (t < after))

        new_data = np.r_[self.at(before),
                         np.stack(self.coords)[index],
                         self.at(after)]

        df = (pd.DataFrame.from_records(np.c_[new_data[:-1,:], new_data[1:,:]],
                                        columns=['lat1', 'lon1', 'alt1',
                                                 'lat2', 'lon2', 'alt2']).
              assign(time1=[before, *t[index]],
                     time2=[*t[index], after],
                     origin=self.origin,
                     destination=self.destination,
                     aircraft=self.aircraft,
                     callsign=self.callsign))

        return Flight(df)




class SO6(object):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, callsign):
        return Flight(self.data.groupby('callsign').get_group(callsign))

    def __iter__(self):
        return iter(self.data.groupby('callsign'))

    def __len__(self):
        return len(self.callsigns)

    def _ipython_key_completions_(self):
        return self.callsigns

    @property
    def callsigns(self):
        return set(self.data.callsign)

    @classmethod
    def parse_so6(self, filename):
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
    def parse_pkl(self, filename):
        so6 = pd.read_pickle(filename)
        return SO6(so6)

    def to_pkl(self, filename):
        self.data.to_pickle(filename)

    def at(self, time):
        time = to_datetime(time)
        return SO6(self.data[(self.data.time1 <= time) &
                             (self.data.time2 > time)])

    def between(self, before, after):
        before, after = to_datetime(before), to_datetime(after)
        return SO6(self.data[(self.data.time1 <= after) &
                             (self.data.time2 >= before)])

    def intersects(self, sector):
        west, south, east, north = sector.bounds
        sub = self.data[(self.data.lat1 <= north) &
                        (self.data.lat2 >= south) &
                        (self.data.lon1 <= east) &
                        (self.data.lon2 >= west)]
        return SO6(sub.groupby('callsign').filter(
            lambda f: sector.intersects(Flight(f).linestring())))




