# fmt: off
import json
import logging
import os
import subprocess
import sys
from calendar import timegm
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import (Dict, Iterable, Iterator, List, NoReturn, Optional, Set,
                    Tuple, Type, TypeVar, Union, overload)

import numpy as np
import pandas as pd
import pyproj
from cartopy.crs import PlateCarree
from scipy.interpolate import interp1d
from shapely.geometry import LineString, base

from ....core import Airspace
from ....core import Flight as FlightMixin
from ....core.flight import Position
from ....core.mixins import DataFrameMixin
from ....core.time import time_or_delta, timelike, to_datetime

# fmt: on


def _prepare_libarchive():  # coverage: ignore
    """
    There are some well documented issues in MacOS about libarchive.
    Let's try to do things ourselves...

    https://github.com/dsoprea/PyEasyArchive
    """

    if sys.platform != "darwin":
        return

    if "LA_LIBRARY_FILEPATH" in os.environ:
        return

    command = ["brew", "info", "--json=v1", "libarchive"]

    try:
        result = subprocess.check_output(command)
    except Exception as e:
        logging.error("Could not lookup 'libarchive' package info", e)

    info = json.loads(result)
    installed_versions = info[0]["installed"]
    if len(installed_versions) == 0:
        logging.warning("libarchive is not currently installed via Brew")
        return

    version = installed_versions[0]["version"]

    command = ["brew", "--cellar", "libarchive"]
    package_path = subprocess.check_output(command)[:-1]

    library_path = os.path.join(package_path.decode(), version, "lib")
    os.environ["LA_LIBRARY_FILEPATH"] = os.path.join(
        library_path, "libarchive.dylib"
    )


# https://github.com/python/mypy/issues/2511
SO6TypeVar = TypeVar("SO6TypeVar", bound="SO6")


def time(int_: int) -> datetime:
    ts = timegm((2000 + int_ // 10000, int_ // 100 % 100, int_ % 100, 0, 0, 0))
    return datetime.fromtimestamp(ts, timezone.utc)


def hour(int_: int) -> timedelta:
    return timedelta(
        hours=int_ // 10000, minutes=int_ // 100 % 100, seconds=int_ % 100
    )


class Flight(FlightMixin):
    """

    SO6 Flight inherit from `traffic.core.Flight </traffic.core.flight.html>`_
    and implement specificities of the SO6 format.

    .. code:: python

        so6['HOP36PP'].data.drop(
            columns=['alt1', 'alt2', 'aircraft', 'callsign', 'flight_id']
        )


    .. raw:: html

        <div>
        <style scoped>
            .dataframe tbody tr th:only-of-type {
                vertical-align: middle;
            }
            .dataframe tbody tr th {
                vertical-align: top;
            }
            .dataframe thead th {
                text-align: right;
            }
        </style>
        <table border="0" class="dataframe">
        <thead>
            <tr style="text-align: right;">
            <th></th>
            <th>origin</th>
            <th>destination</th>
            <th>lat1</th>
            <th>lon1</th>
            <th>lat2</th>
            <th>lon2</th>
            <th>time1</th>
            <th>time2</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <th>65794</th>
            <td>LFML</td>
            <td>LFBD</td>
            <td>43.608398</td>
            <td>4.527325</td>
            <td>44.543555</td>
            <td>1.178150</td>
            <td>2018-01-01 18:15:40+00:00</td>
            <td>2018-01-01 18:44:50+00:00</td>
            </tr>
            <tr>
            <th>65795</th>
            <td>LFML</td>
            <td>LFBD</td>
            <td>44.543555</td>
            <td>1.178150</td>
            <td>44.726898</td>
            <td>0.460837</td>
            <td>2018-01-01 18:44:50+00:00</td>
            <td>2018-01-01 18:52:10+00:00</td>
            </tr>
            <tr>
            <th>65796</th>
            <td>LFML</td>
            <td>LFBD</td>
            <td>44.726898</td>
            <td>0.460837</td>
            <td>44.751343</td>
            <td>-0.091422</td>
            <td>2018-01-01 18:52:10+00:00</td>
            <td>2018-01-01 18:58:00+00:00</td>
            </tr>
        </tbody>
        </table>
        </div>


    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)
        self.interpolator: Dict = dict()

    def __add__(self, other):
        if other == 0:
            # useful for compatibility with sum() function
            return self

        return SO6(pd.concat([self.data, other.data], sort=False))

    @property
    def timestamp(self) -> Iterator[pd.Timestamp]:
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.time1
        yield s.time2

    @property
    def aircraft(self) -> str:
        return self.data.iloc[0].aircraft

    @property
    def typecode(self) -> str:
        return self.data.iloc[0].aircraft

    @property
    def start(self) -> pd.Timestamp:
        return min(self.timestamp)

    @property
    def stop(self) -> pd.Timestamp:
        return max(self.timestamp)

    @property
    def registration(self) -> None:
        return None

    def coords4d(
        self, delta_t: bool = False
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
    def xy_time(self) -> Iterator[Tuple[float, float, float]]:
        if self.data.shape[0] == 0:
            return
        for _, s in self.data.iterrows():
            yield s.lon1, s.lat1, s.time1.to_pydatetime().timestamp()
        yield s.lon2, s.lat2, s.time2.to_pydatetime().timestamp()

    @property
    def linestring(self) -> LineString:
        return LineString(list(self.coords))

    @property
    def shape(self) -> LineString:
        return self.linestring

    def airborne(self) -> "Flight":
        """Identity method, available for consistency"""
        return self

    def interpolate(self, times, proj=PlateCarree()) -> np.ndarray:
        if proj not in self.interpolator:
            self.interpolator[proj] = interp1d(
                np.stack(
                    [t.to_pydatetime().timestamp() for t in self.timestamp]
                ),
                proj.transform_points(
                    PlateCarree(), *np.stack(list(self.coords)).T
                ).T,
            )
        return PlateCarree().transform_points(
            proj, *self.interpolator[proj](times)
        )

    def at(self, time: Optional[timelike] = None) -> Position:
        """
        Since DDR files contain few points on their trajectory, interpolation
        functions are provided:

        .. code:: python

            >>> so6['HOP36PP'].at("2018/01/01 18:40")

            longitude        1.733156
            latitude        44.388586
            altitude     26638.857143
            dtype: float64

        """
        if time is None:
            raise NotImplementedError()

        time = to_datetime(time)
        timearray: np.ndarray[datetime] = np.array([time.timestamp()])
        res = self.interpolate(timearray)

        return Position(
            pd.Series(res[0], index=["longitude", "latitude", "altitude"])
        )

    def between(
        self, start: timelike, stop: time_or_delta, strict: bool = True
    ) -> "Flight":
        """
        WARNING: strict: bool = True is not taken into account yet.
        """
        start = to_datetime(start)
        if isinstance(stop, timedelta):
            stop = start + stop
        else:
            stop = to_datetime(stop)

        t: np.ndarray = np.stack(list(self.timestamp))
        index = np.where((start < t) & (t < stop))

        new_data: np.ndarray = np.stack(list(self.coords))[index]
        time1: List[datetime] = [start, *t[index]]
        time2: List[datetime] = [*t[index], stop]

        if start > t[0]:
            new_data = np.vstack([self.at(start), new_data])
        else:
            time1, time2 = time1[1:], time2[1:]
        if stop < t[-1]:
            new_data = np.vstack([new_data, self.at(stop)])
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

    def clip(self, shape: base.BaseGeometry) -> Optional["Flight"]:
        linestring = LineString(list(self.xy_time))
        intersection = linestring.intersection(shape)
        begin: Optional[datetime] = None

        if intersection.is_empty:
            return None

        if isinstance(intersection, LineString):
            begin, *_, end = list(
                datetime.fromtimestamp(t, timezone.utc)
                for t in np.stack(intersection.coords)[:, 2]
            )

        else:
            for x in LineString(list(self.xy_time)).intersection(shape):
                begin_, *_, end = list(
                    datetime.fromtimestamp(t, timezone.utc)
                    for t in np.stack(x.coords)[:, 2]
                )
                if begin is None:
                    begin = begin_

        return self.between(begin, end)

    def clip_altitude(self, min_: int, max_: int) -> Iterator["Flight"]:
        """
        Splits a Flight in several segments comprised between altitudes
        `min_` and `max_`, with proper interpolations where needed.
        """

        def buffer_to_iter(
            proj: pyproj.Proj, buffer: List[pd.Series]
        ) -> Iterator["Flight"]:
            df = pd.DataFrame.from_records(buffer)

            df["lon1"], df["lat1"] = pyproj.transform(
                proj, pyproj.Proj(init="EPSG:4326"), df.x1.values, df.y1.values
            )

            df["lon2"], df["lat2"] = pyproj.transform(
                proj, pyproj.Proj(init="EPSG:4326"), df.x2.values, df.y2.values
            )

            yield self.__class__(df.drop(["x1", "x2", "y1", "y2"], axis=1))

        data = self.data.copy()

        proj = pyproj.Proj(
            proj="lcc",
            ellps="WGS84",
            lat_0=data.lat1.mean(),
            lon_0=data.lon1.mean(),
            lat_1=data.lat1.min(),
            lat_2=data.lat1.max(),
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

    def resample(self) -> NoReturn:  # type: ignore
        """
        The resampling method is not available.
        """
        raise NotImplementedError


class SO6(DataFrameMixin):
    """
    Eurocontrol DDR files provide flight intentions of aircraft for a full
    day of traffic across Europe. This data cannot be shared so the file
    included in the repository has actually been generated from OpenSky
    ADS-B data to match so6 format.

    `SO6 <#traffic.data.SO6>`_ represent a collection of trajectories, the
    bracket notation yields a `Flight <#traffic.data.so6.so6.Flight>`_
    structure adapted to the specificities of the SO6 format.

    .. code:: python

        from traffic.data import SO6
        so6 = SO6.from_file("./data/sample_m3.so6.7z")
        so6.to_pickle("./data/sample_m3.pkl")

    If you are going to work a lot with data for one day, it is recommended
    to serialize the data so that it loads faster. The structure holds a
    DataFrame in the data attribute.

    You can then access data from the so6 file, by flight, with the bracket
    notation. Interactive environments (IPython, Jupyter notebooks) provide
    completion on the flight names.

    Callsigns may not be enough to discriminate flights because of same
    callsigns assigned to a trip with many legs. In general, you can access a
    Flight from its unique ID or from its callsign

    .. code:: python

        so6[332206265]
        # here equivalent to: so6['HOP36PP']

    .. raw:: html

        <b>Flight HOP36PP</b> (332206265)<ul>
        <li><b>aircraft:</b> A319</li>
        <li><b>origin:</b> LFML (2018-01-01 18:15:40+00:00)</li>
        <li><b>destination:</b> LFBD (2018-01-01 18:58:00+00:00)</li>
        </ul>
        <div style="white-space: nowrap">
        <svg xmlns="http://www.w3.org/2000/svg"
         xmlns:xlink="http://www.w3.org/1999/xlink"
         width="300" height="300"
         viewBox="-22064.364032842677 4643541.548496112
         400649.87556558463 148424.4619210167"
         preserveAspectRatio="xMinYMin meet">
         <g transform="matrix(1,0,0,-1,0,9435507.558913242)">
         <polyline fill="none" stroke="#66cc99" stroke-width="2670.999170437231"
         points="363746.62725253514,4658380.432776319 93398.87407311927,
         4754561.883957243 36435.06118046089,4774490.218033796
         -7225.479752635839,4777127.126136922" opacity="0.8" />
         </g></svg></div>

    """

    __slots__ = ("data",)

    identifier = Union[int, str]

    @overload
    def __getitem__(self, index: identifier) -> Flight:
        ...

    @overload  # noqa: F811
    def __getitem__(
        self, index: Union["SO6", Set["str"], Set[int]]
    ) -> Optional["SO6"]:
        ...

    def __getitem__(  # noqa: F811
        self, index: Union[identifier, "SO6", Set["str"], Set[int]]
    ) -> Union[Flight, "SO6", None]:

        if isinstance(index, int):
            return Flight(self.data.groupby("flight_id").get_group(index))
        if isinstance(index, str):
            return Flight(self.data.groupby("callsign").get_group(index))

        if isinstance(index, SO6):
            # not very natural, but why not...
            index = index.flight_ids
        list_query = list(index)

        if len(list_query) == 0:
            return None

        if isinstance(list_query[0], str):
            select = self.data.callsign.isin(list_query)
        else:
            select = self.data.flight_id.isin(list_query)

        return SO6(self.data[select])

    def __iter__(self) -> Iterator[Flight]:
        for _, flight in self.data.groupby("flight_id"):
            yield Flight(flight)

    def __len__(self) -> int:
        return len(self.flight_ids)

    def _ipython_key_completions_(self):
        return {*self.flight_ids, *self.callsigns}

    def __add__(self, other: Union[Flight, "SO6"]) -> "SO6":
        # useful for compatibility with sum() function
        if other == 0:
            return self
        return self.__class__(pd.concat([self.data, other.data], sort=False))

    def __radd__(self, other: Union[Flight, "SO6"]) -> "SO6":
        return self + other

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

    @lru_cache()
    def stats(self) -> pd.DataFrame:  # coverage: ignore
        cumul = []
        for f in self:
            info = {
                "flight_id": f.flight_id,
                "callsign": f.callsign,
                "origin": f.origin,
                "destination": f.destination,
                "duration": f.stop - f.start,
            }
            cumul.append(info)

        return (
            pd.DataFrame.from_records(cumul)
            .set_index("flight_id")
            .sort_values("duration", ascending=False)
        )

    def __repr__(self) -> str:
        stats = self.stats()
        return stats.__repr__()

    def _repr_html_(self) -> str:
        stats = self.stats()
        return stats._repr_html_()

    @classmethod
    def from_so6(
        cls: Type[SO6TypeVar], filename: Union[str, Path, StringIO]
    ) -> SO6TypeVar:
        so6 = pd.read_csv(
            filename,
            sep=" ",
            header=None,
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

        return cls(so6)

    @classmethod
    def from_so6_7z(
        cls: Type[SO6TypeVar], filename: Union[str, Path]
    ) -> SO6TypeVar:
        _prepare_libarchive()
        from libarchive.public import memory_reader

        with open(filename, "rb") as fh:
            with memory_reader(fh.read()) as entries:
                s = StringIO()
                for file in entries:
                    for block in file.get_blocks():
                        s.write(block.decode())
                s.seek(0)
                so6 = cls.from_so6(s)
                s.close()
                return so6

    @classmethod
    def from_file(
        cls: Type[SO6TypeVar], filename: Union[Path, str], **kwargs
    ) -> Optional[SO6TypeVar]:  # coverage: ignore
        """
        In addition to `usual formats
        <export.html#traffic.core.mixins.DataFrameMixin>`_, you can parse so6
        files as text files (.so6 extension) or as 7-zipped text files (.so6.7z
        extension).

        .. warning::

            You will need the `libarchive
            <https://github.com/dsoprea/PyEasyArchive>`_ library to be able
            to parse .so6.7z files on the fly.

        """
        path = Path(filename)
        if path.suffixes == [".so6", ".7z"]:
            return cls.from_so6_7z(filename)
        if path.suffixes == [".so6"]:
            return cls.from_so6(filename)
        return super().from_file(filename)

    def at(self, time: timelike) -> "SO6":
        """Selects all segments of the SO6 dataframe with ``time`` included
        in the segment interval.
        """
        time = to_datetime(time)
        return SO6(
            self.data[(self.data.time1 <= time) & (self.data.time2 > time)]
        )

    def between(self, start: timelike, stop: time_or_delta) -> "SO6":
        """Selects all segments of the SO6 dataframe with intervals intersecting
        [``start``, ``stop``].

        The ``stop`` argument may be also be written as a
        ``datetime.timedelta``.
        """
        start = to_datetime(start)
        if isinstance(stop, timedelta):
            stop = start + stop
        else:
            stop = to_datetime(stop)
        return SO6(
            self.data[(self.data.time1 <= stop) & (self.data.time2 >= start)]
        )

    def intersects(self, sector: Airspace) -> "SO6":
        """
        Selects all Flights intersecting the given airspace.

        .. tip::

            See the impact of calling `.inside_bbox(sector)
            <#traffic.data.SO6.inside_bbox>`_ to the bounding box
            before intersecting the airspace. Note that this
            chaining may not be safe for smaller airspaces.

        .. code:: python

            noon = so6.at("2018/01/01 12:00")

        .. code:: python

            %%time
            bdx_flights = noon.intersects(nm_airspaces['LFBBBDX'])

            CPU times: user 3.9 s, sys: 0 ns, total: 3.9 s
            Wall time: 3.9 s


        .. code:: python

            %%time
            bdx_flights = (
                noon
                .inside_bbox(nm_airspaces["LFBBBDX"])
                .intersects(nm_airspaces["LFBBBDX"])
            )

            CPU times: user 1.42 s, sys: 8.27 ms, total: 1.43 s
            Wall time: 1.43 s
        """
        return SO6(
            self.data.groupby("flight_id").filter(
                lambda flight: Flight(flight).intersects(sector)
            )
        )

    def inside_bbox(self, bounds: Union[Airspace, Tuple[float, ...]]) -> "SO6":
        """
        Selects all Flights intersecting the bounding box of the given airspace.

        A tuple (west, south, east, north) is also accepted as a parameter.

        .. code:: python

            >>> bdx_so6 = so6.inside_bbox(nm_airspaces["LFBBBDX"])
            >>> f"before: {len(so6)} flights, after: {len(bdx_so6)} flights"
            before: 11043 flights, after: 1548 flights

        """

        if isinstance(bounds, Airspace):
            bounds = bounds.flatten().bounds

        if isinstance(bounds, base.BaseGeometry):
            bounds = bounds.bounds

        west, south, east, north = bounds

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
