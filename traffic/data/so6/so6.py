import warnings
from calendar import timegm
from datetime import datetime, timedelta
# from decimal import ROUND_DOWN, ROUND_UP, Decimal
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

import pandas as pd
import pyproj
from cartopy.crs import PlateCarree
from scipy.interpolate import interp1d
from shapely.geometry import LineString, base

from ...core import Flight as FlightMixin, Airspace
from ...core.mixins import DataFrameMixin
from ...core.time import time_or_delta, timelike, to_datetime


def time(int_: int) -> datetime:
    ts = timegm((2000 + int_ // 10000, int_ // 100 % 100, int_ % 100, 0, 0, 0))
    return datetime.fromtimestamp(ts)


def hour(int_: int) -> timedelta:
    return timedelta(
        hours=int_ // 10000, minutes=int_ // 100 % 100, seconds=int_ % 100
    )


class Flight(FlightMixin):

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
        self.interpolator: Dict = dict()

    @property
    def timestamp(self) -> Iterator[datetime]:
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.time1
        yield s.time2

    @property
    def aircraft(self) -> str:
        return self.data.iloc[0].aircraft

    @property
    def registration(self) -> None:
        return None

    def coords4d(
            self, delta_t: bool=False
    ) -> Iterator[Tuple[float, float, float, float]]:
        t = 0
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            if delta_t:
                yield t, s.lon1, s.lat1, s.alt1
                t += (s.time2 - s.time1).total_seconds()
            else:
                yield s.time1, s.lon1, s.lat1, s.alt1
        if delta_t:
            yield t, s.lon2, s.lat2, s.alt2
        else:
            yield s.time2, s.lon2, s.lat2, s.alt2

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.lon1, s.lat1, s.alt1
        yield s.lon2, s.lat2, s.alt2

    @property
    def linestring(self) -> LineString:
        return LineString(list(self.coords))

    @property
    def shape(self) -> LineString:
        return self.linestring

    def airborne(self):
        return self

    def interpolate(self, times, proj=PlateCarree()):
        """Interpolates a trajectory in time.  """
        if proj not in self.interpolator:
            self.interpolator[proj] = interp1d(
                np.stack(t.to_pydatetime().timestamp() for t in self.timestamp),
                proj.transform_points(
                    PlateCarree(), *np.stack(self.coords).T
                ).T,
            )
        return PlateCarree().transform_points(
            proj, *self.interpolator[proj](times)
        )

    def at(self, time: timelike, proj=PlateCarree()) -> np.ndarray:
        time = to_datetime(time)
        timearray: np.ndarray[datetime] = np.array([time.timestamp()])
        return self.interpolate(timearray, proj)

    def between(self, before: timelike, after: time_or_delta) -> "Flight":
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)

        t: np.ndarray = np.stack(self.timestamp)
        index = np.where((before < t) & (t < after))

        new_data: np.ndarray = np.stack(self.coords)[index]
        time1: List[datetime] = [before, *t[index]]
        time2: List[datetime] = [*t[index], after]

        if before > t[0]:
            new_data = np.vstack([self.at(before), new_data])
        else:
            time1, time2 = time1[1:], time2[1:]
        if after < t[-1]:
            new_data = np.vstack([new_data, self.at(after)])
        else:
            time1, time2 = time1[:-1], time2[:-1]

        df: pd.DataFrame = (
            pd.DataFrame.from_records(
                np.c_[new_data[:-1, :], new_data[1:, :]],
                columns=["lon1", "lat1", "alt1", "lon2", "lat2", "alt2"],
            ).assign(
                time1=time1,
                time2=time2,
                origin=self.origin,
                destination=self.destination,
                aircraft=self.aircraft,
                flight_id=self.flight_id,
                callsign=self.callsign,
            )
        )

        return Flight(df)

    def clip(self, shape: base.BaseGeometry) -> "Flight":
        coords = np.stack(self.linestring.intersection(shape).coords)
        times = list(
            datetime.fromtimestamp(t)
            for t in np.stack(
                LineString(list(self.xy_time)).intersection(shape).coords
            )[:, 2]
        )

        df: pd.DataFrame = (
            pd.DataFrame.from_records(
                np.c_[coords[:-1, :], coords[1:, :]],
                columns=["lon1", "lat1", "alt1", "lon2", "lat2", "alt2"],
            ).assign(
                time1=times[:-1],
                time2=times[1:],
                origin=self.origin,
                destination=self.destination,
                aircraft=self.aircraft,
                flight_id=self.flight_id,
                callsign=self.callsign,
            )
        )
        return Flight(df)

    def clip_altitude(self, min_: int, max_: int) -> Iterator["Flight"]:
        def buffer_to_iter(proj, buffer):
            df = pd.DataFrame.from_records(buffer)

            df["lon1"], df["lat1"] = pyproj.transform(
                proj, pyproj.Proj(init="EPSG:4326"), df.x1.values, df.y1.values
            )

            df["lon2"], df["lat2"] = pyproj.transform(
                proj, pyproj.Proj(init="EPSG:4326"), df.x2.values, df.y2.values
            )

            yield df.drop(["x1", "x2", "y1", "y2"], axis=1)

        data = self.data.copy()

        proj = pyproj.Proj(
            proj="lcc",
            lat0=data.lat1.mean(),
            lon0=data.lon1.mean(),
            lat1=data.lat1.min(),
            lat2=data.lat1.max(),
        )

        data["x1"], data["y1"] = pyproj.transform(
            pyproj.Proj(init="EPSG:4326"),
            proj,
            data.lon1.values,
            data.lat1.values,
        )

        data["x2"], data["y2"] = pyproj.transform(
            pyproj.Proj(init="EPSG:4326"),
            proj,
            data.lon2.values,
            data.lat2.values,
        )

        buffer = []
        for (_, line) in data.iterrows():
            if (line.alt1 < max_ or line.alt2 < max_) and (
                line.alt1 > min_ or line.alt2 > min_
            ):

                if line.alt1 != line.alt2:
                    f_x = (line.x1 - line.x2) / (line.alt1 - line.alt2)
                    f_y = (line.y1 - line.y2) / (line.alt1 - line.alt2)

                if line.alt1 > max_:
                    line.x1 = line.x2 + (max_ - line.alt2) * f_x
                    line.y1 = line.y2 + (max_ - line.alt2) * f_y
                    line.alt1 = max_
                if line.alt1 < min_:
                    line.x1 = line.x2 + (min_ - line.alt2) * f_x
                    line.y1 = line.y2 + (min_ - line.alt2) * f_y
                    line.alt1 = min_
                if line.alt2 > max_:
                    line.x2 = line.x1 + (max_ - line.alt1) * f_x
                    line.y2 = line.y1 + (max_ - line.alt1) * f_y
                    line.alt2 = max_
                if line.alt2 < min_:
                    line.x2 = line.x1 + (min_ - line.alt1) * f_x
                    line.y2 = line.y1 + (min_ - line.alt1) * f_y
                    line.alt2 = min_

                buffer.append(line)

            else:
                if len(buffer) > 0:
                    yield from buffer_to_iter(proj, buffer)
                    buffer = []

        if len(buffer) > 0:
            yield from buffer_to_iter(proj, buffer)

    def resample(self):
        raise NotImplementedError("SO6 do not provide a resample method")


class SO6(DataFrameMixin):

    identifier = Union[int, str]

    def __getitem__(self, _id: identifier) -> Flight:
        if isinstance(_id, int):
            return Flight(self.data.groupby("flight_id").get_group(_id))
        if isinstance(_id, str):
            return Flight(self.data.groupby("callsign").get_group(_id))

    def __iter__(self) -> Iterator[Flight]:
        for _, flight in self.data.groupby("flight_id"):
            yield Flight(flight)

    def __len__(self) -> int:
        return len(self.flight_ids)

    def _ipython_key_completions_(self):
        return {*self.flight_ids, *self.callsigns}

    def get(self, callsign: str) -> Iterable[Tuple[int, Flight]]:
        all_flights = self.data.groupby("callsign").get_group(callsign)
        for flight_id, flight in all_flights.groupby("flight_id"):
            yield flight_id, Flight(flight)

    @property
    def start_time(self) -> pd.Timestamp:
        return min(self.data.time1)

    @property
    def end_time(self) -> pd.Timestamp:
        return max(self.data.time2)

    @property
    def callsigns(self) -> Set[str]:
        return set(self.data.callsign)

    @property
    def flight_ids(self) -> Set[int]:
        return set(self.data.flight_id)

    @classmethod
    def from_so6(self, filename: Union[str, Path, StringIO]) -> "SO6":
        so6 = pd.read_csv(
            filename,
            sep=" ",
            header=-1,
            names=[
                "d1",
                "origin",
                "destination",
                "aircraft",
                "hour1",
                "hour2",
                "alt1",
                "alt2",
                "d2",
                "callsign",
                "date1",
                "date2",
                "lat1",
                "lon1",
                "lat2",
                "lon2",
                "flight_id",
                "d3",
                "d4",
                "d5",
            ],
        )

        so6 = so6.assign(
            lat1=so6.lat1 / 60,
            lat2=so6.lat2 / 60,
            lon1=so6.lon1 / 60,
            lon2=so6.lon2 / 60,
            alt1=so6.alt1 * 100,
            alt2=so6.alt2 * 100,
            time1=so6.date1.apply(time) + so6.hour1.apply(hour),
            time2=so6.date2.apply(time) + so6.hour2.apply(hour),
        )

        for col in (
            "d1",
            "d2",
            "d3",
            "d4",
            "d5",
            "date1",
            "date2",
            "hour1",
            "hour2",
        ):
            del so6[col]

        return SO6(so6)

    @classmethod
    def from_so6_7z(self, filename: Union[str, Path]) -> "SO6":
        from libarchive.public import memory_reader

        with open(filename, "rb") as fh:
            with memory_reader(fh.read()) as entries:
                s = StringIO()
                for file in entries:
                    for block in file.get_blocks():
                        s.write(block.decode())
                s.seek(0)
                so6 = SO6.from_so6(s)
                s.close()
                return so6

    @classmethod
    def parse_file(cls, filename: str) -> Optional["SO6"]:
        warnings.warn("Use SO6.from_file(filename)", DeprecationWarning)
        return cls.from_file(filename)

    @classmethod
    def from_file(cls, filename: Union[Path, str]) -> Optional["SO6"]:
        path = Path(filename)
        if path.suffixes == [".so6", ".7z"]:
            return SO6.from_so6_7z(filename)
        if path.suffixes == [".so6"]:
            return SO6.from_so6(filename)
        holder = DataFrameMixin.from_file(filename)
        return SO6(holder.data) if holder is not None else None

    def at(self, time: timelike) -> "SO6":
        time = to_datetime(time)
        return SO6(
            self.data[(self.data.time1 <= time) & (self.data.time2 > time)]
        )

    def between(self, before: timelike, after: time_or_delta) -> "SO6":
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)
        return SO6(
            self.data[(self.data.time1 <= after) & (self.data.time2 >= before)]
        )

    def intersects(self, sector: Airspace) -> "SO6":
        return SO6(
            self.data.groupby("flight_id").filter(
                lambda flight: Flight(flight).intersects(sector)  # type:ignore
            )
        )

    def inside_bbox(self, bounds: Union[Airspace, Tuple[float, ...]]) -> "SO6":

        if isinstance(bounds, Airspace):
            bounds = bounds.flatten().bounds

        if isinstance(bounds, base.BaseGeometry):
            bounds = bounds.bounds

        west, south, east, north = bounds

        # Transform coords into intelligible floats
        # '-2.06 <= lon1 <= 4.50 & 42.36 <= lat1 <= 48.14', instead of
        #  (-2.066666603088379, 42.366943359375, 4.491666793823242,
        #   48.13333511352539)
        # dec = Decimal('0.00')
        # west = Decimal(west).quantize(dec, rounding=ROUND_DOWN)
        # east = Decimal(east).quantize(dec, rounding=ROUND_UP)
        # south = Decimal(south).quantize(dec, rounding=ROUND_DOWN)
        # north = Decimal(north).quantize(dec, rounding=ROUND_UP)

        # the numexpr query is 10% faster than the regular
        # data[data.lat1 >= ...] conjunctions of comparisons
        query = "{0} <= lon1 <= {2} and {1} <= lat1 <= {3}"
        query = query.format(west, south, east, north)

        data = self.data.query(query)

        callsigns: Set[str] = set(data.callsign)

        return SO6(
            self.data.groupby("flight_id").filter(
                lambda data: data.iloc[0].callsign in callsigns
            )
        )

    def select(self, query: Union["SO6", Iterable[str]]) -> "SO6":
        if isinstance(query, SO6):
            # not very natural, but why not...
            query = query.callsigns
        select = self.data.callsign.isin(query)
        return SO6(self.data[select])
