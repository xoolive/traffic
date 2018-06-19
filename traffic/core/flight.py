import logging
from datetime import datetime, timedelta
from typing import Dict, Iterator, Optional, Set, Tuple, Union

import numpy as np

import pandas as pd
import pyproj
from shapely.geometry import LineString, base

from ..core.time import time_or_delta, timelike, to_datetime
from .mixins import DataFrameMixin, GeographyMixin, ShapelyMixin


def split(data: pd.DataFrame, value, unit) -> Iterator[pd.DataFrame]:
    diff = data.timestamp.diff().values
    if diff.max() > np.timedelta64(value, unit):
        yield from split(data.iloc[: diff.argmax()], value, unit)
        yield from split(data.iloc[diff.argmax() :], value, unit)
    else:
        yield data


class Flight(DataFrameMixin, ShapelyMixin, GeographyMixin):
    def __add__(self, other):
        # useful for compatibility with sum() function
        if other == 0:
            return self
        # keep import here to avoid recursion
        from .traffic import Traffic

        return Traffic.from_flights([self, other])

    def __radd__(self, other):
        return self + other

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
            title += f"({self.stop})</li>"
        else:
            title += f"<li><b>destination:</b> {self.stop}</li>"
        title += "</ul>"
        return title

    def _repr_html_(self):
        title = self.info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    @property
    def timestamp(self) -> Iterator[datetime]:
        yield from self.data.timestamp

    @property
    def start(self) -> datetime:
        return min(self.timestamp)

    @property
    def stop(self) -> datetime:
        return max(self.timestamp)

    @property
    def callsign(self) -> Union[str, Set[str]]:
        tmp = set(self.data.callsign)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several callsigns for one flight, consider splitting")
        return tmp

    @property
    def number(self) -> Optional[Union[str, Set[str]]]:
        if "number" not in self.data.columns:
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
        if "flight_id" not in self.data.columns:
            return None
        tmp = set(self.data.flight_id)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several ids for one flight, consider splitting")
        return tmp

    @property
    def origin(self) -> Optional[Union[str, Set[str]]]:
        if "origin" not in self.data.columns:
            return None
        tmp = set(self.data.origin)
        if len(tmp) == 1:
            return tmp.pop()
        logging.warn("Several origins for one flight, consider splitting")
        return tmp

    @property
    def destination(self) -> Optional[Union[str, Set[str]]]:
        if "destination" not in self.data.columns:
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
    def registration(self) -> Optional[str]:
        from ..data import aircraft as acdb

        if not isinstance(self.icao24, str):
            return None
        ac = acdb[self.icao24]
        if ac.shape[0] != 1:
            return None
        return ac.iloc[0].regid

    @property
    def coords(self) -> Iterator[Tuple[float, float, float]]:
        data = self.data[self.data.longitude.notnull()]
        altitude = (
            "baro_altitude"
            if "baro_altitude" in self.data.columns
            else "altitude"
        )
        yield from zip(data["longitude"], data["latitude"], data[altitude])

    @property
    def xy_time(self) -> Iterator[Tuple[float, float, float]]:
        iterator = iter(zip(self.coords, self.timestamp))
        while True:
            next_ = next(iterator, None)
            if next_ is None:
                return
            coords, time = next_
            yield (coords[0], coords[1], time.timestamp())

    @property
    def linestring(self) -> Optional[LineString]:
        coords = list(self.coords)
        if len(coords) < 2:
            return None
        return LineString(coords)

    @property
    def shape(self) -> Optional[LineString]:
        return self.linestring

    def airborne(self) -> "Flight":
        altitude = (
            "baro_altitude"
            if "baro_altitude" in self.data.columns
            else "altitude"
        )
        return self.__class__(self.data[self.data[altitude].notnull()])

    # -- Interpolation and resampling --

    def split(self, value: int = 10, unit: str = "m") -> Iterator["Flight"]:
        for data in split(self.data, value, unit):
            yield self.__class__(data)

    def resample(self, rule: str = "1s") -> "Flight":
        data = (
            self.data.assign(start=self.start, stop=self.stop)
            .set_index("timestamp")
            .resample(rule)
            .interpolate()
            .reset_index()
            .fillna(method="pad")
        )
        return self.__class__(data)

    def at(self, time: timelike) -> pd.core.series.Series:
        index = to_datetime(time)
        return self.data.set_index("timestamp").loc[time]

    def between(self, before: timelike, after: time_or_delta) -> "Flight":
        before = to_datetime(before)
        if isinstance(after, timedelta):
            after = before + after
        else:
            after = to_datetime(after)

        t: np.ndarray = np.stack(self.timestamp)
        index = np.where((before < t) & (t < after))
        return self.__class__(self.data.iloc[index])

    # -- Geometry operations --

    def clip(
        self, shape: base.BaseGeometry
    ) -> Union[None, "Flight", Iterator["Flight"]]:


        linestring = LineString(list(self.airborne().xy_time))
        intersection = linestring.intersection(shape)

        if intersection.is_empty:
            return None

        if isinstance(intersection, LineString):
            times = list(
                datetime.fromtimestamp(t)
                for t in np.stack(intersection.coords)[:, 2]
            )
            return self.__class__(
                self.data[
                    (self.data.timestamp >= min(times))
                    & (self.data.timestamp <= max(times))
                ]
            )

        def _clip_generator():
            for segment in intersection:
                times = list(
                    datetime.fromtimestamp(t)
                    for t in np.stack(segment.coords)[:, 2]
                )
                yield self.__class__(
                    self.data[
                        (self.data.timestamp >= min(times))
                        & (self.data.timestamp <= max(times))
                    ]
                )

        return (leg for leg in _clip_generator())

    # -- Visualisation --

    def plot(self, ax, **kwargs):
        if "projection" in ax.__dict__ and "transform" not in kwargs:
            from cartopy.crs import PlateCarree

            kwargs["transform"] = PlateCarree()
        ax.plot(*self.shape.xy, **kwargs)
