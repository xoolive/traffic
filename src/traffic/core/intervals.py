from __future__ import annotations

import datetime
from typing import Any, Iterator, Literal, overload

import numpy as np
import pandas as pd

from .mixins import DataFrameMixin
from .time import timelike, to_datetime


class Interval:
    start: datetime.datetime
    stop: datetime.datetime

    def __init__(self, start: timelike, stop: timelike) -> None:
        self.start = to_datetime(start)
        self.stop = to_datetime(stop)
        if self.start > self.stop:
            raise RuntimeError("Start value should be anterior to stop value")

    def __repr__(self) -> str:
        return f"[{self.start}, {self.stop}]"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Interval):
            return False
        return self.start == other.start and self.stop == other.stop

    def __req__(self, other: Any) -> bool:
        if not isinstance(other, Interval):
            return False
        return self.start == other.start and self.stop == other.stop

    def duration(self) -> pd.Timedelta:
        return self.stop - self.start

    def overlap(self, other: Interval) -> bool:
        """Returns True if two intervals overlap."""
        return self.start < other.stop and self.stop > other.start

    def __radd__(self, other: Literal[0]) -> IntervalCollection:
        if other == 0:
            return IntervalCollection(self)

    def __add__(self, other: Interval) -> IntervalCollection:
        """Concatenates the two elements in an IntervalCollection."""
        if isinstance(other, Interval):
            return IntervalCollection(
                start=[self.start, other.start],
                stop=[self.stop, other.stop],
            )

        return NotImplemented

    def __sub__(self, other: Interval) -> None | IntervalCollection:
        if isinstance(other, Interval):
            # reduces interval or splits or deletes (returns none) self - other
            if self.start > other.stop or self.stop < other.start:
                return IntervalCollection(start=self.start, stop=self.stop)
            else:
                if self.start < other.start and self.stop > other.stop:
                    return IntervalCollection(
                        start=[self.start, other.stop],
                        stop=[other.start, self.stop],
                    )
                elif self.start < other.start:
                    return IntervalCollection(self.start, other.start)
                elif self.stop > other.stop:
                    return IntervalCollection(other.stop, self.stop)
                else:
                    return None

        return NotImplemented

    def union(self, other: Interval | IntervalCollection) -> IntervalCollection:
        return self | other

    def __or__(self, other: Interval) -> IntervalCollection:
        """extends interval or adds new line if no overlap"""

        if isinstance(other, Interval):
            if self.start > other.stop or self.stop < other.start:
                return IntervalCollection(
                    start=[self.start, other.start],
                    stop=[self.stop, other.stop],
                )
            else:
                start = min(self.start, other.start)
                stop = max(self.stop, other.stop)
                return IntervalCollection(start=start, stop=stop)

        return NotImplemented

    @overload
    def intersection(self, other: Interval) -> None | Interval: ...

    @overload
    def intersection(
        self, other: IntervalCollection
    ) -> None | IntervalCollection: ...

    def intersection(
        self, other: Interval | IntervalCollection
    ) -> None | Interval | IntervalCollection:
        return self & other

    def __and__(self, other: Interval) -> None | Interval:
        """returns the time that exists in both intervals"""

        if isinstance(other, Interval):
            if self.overlap(other):
                start = max(self.start, other.start)
                stop = min(self.stop, other.stop)
                return Interval(start, stop)
            return None

        return NotImplemented


class IntervalCollection(DataFrameMixin):
    """A class to represent collections of Intervals.

    An :class:`~Interval` consists of a start and stop attributes.
    Collections of intervals are stored as a :class:`~pandas.DataFrame`.

    Intervals can be created using one of the following syntaxes:

    >>> sample_dates = pd.date_range("2023-01-01", "2023-02-01", freq="1D")
    >>> t0, t1, t2, t3, *_ = sample_dates

    - as a list of :class:`~Interval`:

        >>> IntervalCollection([Interval(t0, t1), Interval(t2, t3)])
        [[2023-01-01 00:00:00, 2023-01-02 00:00:00], ...]

    - as an expanded tuple of :class:`~Interval`:

        >>> IntervalCollection(Interval(t0, t1), Interval(t2, t3))
        [[2023-01-01 00:00:00, 2023-01-02 00:00:00], ...]

    - a list of start and stop values:

        >>> IntervalCollection([t0, t2], [t1, t3])
        [[2023-01-01 00:00:00, 2023-01-02 00:00:00], ...]

    - as a :class:`~pandas.DataFrame`:

        >>> df = pd.DataFrame({'start': [t0, t2], 'stop': [t1, t3]})
        >>> IntervalCollection(df)
        [[2023-01-01 00:00:00, 2023-01-02 00:00:00], ...]

    """

    data: pd.DataFrame

    def __init__(
        self,
        data: None
        | pd.DataFrame
        | Interval
        | list[Interval]
        | timelike
        | list[timelike] = None,
        *other: Interval | timelike | list[timelike],
        start: None | timelike | list[timelike] = None,
        stop: None | timelike | list[timelike] = None,
    ) -> None:
        if isinstance(data, Interval):
            data = [data, *other]
        if isinstance(data, list):
            if all(isinstance(elt, Interval) for elt in data):
                start = [elt.start for elt in data]
                stop = [elt.stop for elt in data]
                data = None
        if not isinstance(data, pd.DataFrame):
            # We reorder parameters here to accept notations like
            # IntervalCollection(start, stop)
            if start is None or stop is None:
                start, stop, *_ = data, *other, start, stop
                data = None
        if data is None:
            if start is None or stop is None:
                msg = "If no data is specified, provide start and stop"
                raise TypeError(msg)
            if isinstance(start, (str, float, datetime.datetime, pd.Timestamp)):
                start = [start]
                stop = [stop]
            assert isinstance(start, list) and isinstance(stop, list)
            if len(start) == 0 or len(stop) == 0:
                msg = "If no data is specified, provide start and stop"
                raise TypeError(msg)

            data = pd.DataFrame(
                {
                    "start": [to_datetime(t) for t in start],
                    "stop": [to_datetime(t) for t in stop],
                }
            )

        # assert isinstance(data, pd.DataFrame)
        # assert data.eval("(start > stop).sum()") == 0

        self.data = data

    def __iter__(self) -> Iterator[Interval]:
        for _, line in self.data.iterrows():
            yield Interval(line.start, line.stop)

    def __repr__(self) -> str:
        return repr(list(i for i in self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, IntervalCollection):
            if self.data.shape != other.data.shape:
                return False
            left = self.data.sort_values(by=["start"], ignore_index=True)
            right = other.data.sort_values(by=["start"], ignore_index=True)
            return left.equals(right)  # type: ignore
        return False

    def total_duration(self) -> pd.Timedelta:
        """Returns the sum of durations of all intervals."""
        return self.consolidate().data.eval("(stop - start).sum()")

    def __radd__(self, other: Literal[0] | Interval) -> IntervalCollection:
        if other == 0:
            return self
        return IntervalCollection(other) + self

    def __add__(
        self, other: Interval | IntervalCollection
    ) -> IntervalCollection:
        if isinstance(other, Interval):
            other = IntervalCollection(other)

        if isinstance(other, IntervalCollection):
            return IntervalCollection(
                pd.concat([self.data, other.data], ignore_index=True)
            )

        return NotImplemented

    def __rsub__(self, other: Interval) -> None | IntervalCollection:
        return IntervalCollection(other) - self

    def __sub__(
        self, other: Interval | IntervalCollection
    ) -> None | IntervalCollection:
        if isinstance(other, Interval):
            cumul = []
            result = self.consolidate()
            for segment in result:
                if segment.overlap(other):
                    starts_after = other.start > segment.start
                    ends_before = other.stop < segment.stop
                    if starts_after:
                        cumul.append(Interval(segment.start, other.start))
                    if ends_before:
                        cumul.append(Interval(other.stop, segment.stop))
                else:
                    cumul.append(segment)

            if len(cumul) == 0:
                return None

            return sum(interval for interval in cumul)  # type: ignore

        if isinstance(other, IntervalCollection):
            result = self.consolidate()
            for interval in other.consolidate():
                result = result - interval  # type: ignore
                if result is None:
                    return None
            return result

        return NotImplemented

    def consolidate(self) -> IntervalCollection:
        """Consolidate the IntervalCollection.

        >>> sample_dates = pd.date_range("2023-01-01", "2023-02-01", freq="1D")
        >>> t0, t1, t2, t3, *_ = sample_dates
        >>> interval = IntervalCollection([Interval(t0, t2), Interval(t1, t3)])
        >>> interval
        [[2023-01-01 ..., 2023-01-03 ...], [2023-01-02 ..., 2023-01-04 ...]]
        >>> interval.consolidate()
        [[2023-01-01 ..., 2023-01-04 ...]]

        """

        if self.data.shape[0] == 1:
            return self

        zero = np.timedelta64(0)
        if not np.any(self.data.stop.shift() - self.data.start >= zero):
            return self

        # The algorithms proceeds with a swiping line starting from the minimum
        # start value, and builds up a connected interval.

        cumul = []

        start_idx = self.data["start"].idxmin()
        start = self.data.iloc[start_idx]
        swiping_line = start["start"]
        horizon = start["stop"]

        while not pd.isnull(swiping_line):
            # look for the first stop value
            # Note that it may not be the first one we meet
            candidates = self.data.query("@swiping_line <= start <= @horizon")
            # update the horizon
            horizon = candidates["stop"].max()
            candidates = self.data.query("start <= @horizon < stop")
            while candidates.shape[0] > 0:
                horizon = candidates["stop"].max()
                candidates = self.data.query("start <= @horizon < stop")

            cumul.append(Interval(swiping_line, horizon))

            # start a new swiping line
            start = self.data.query("@horizon < start").min()
            swiping_line = start["start"]
            horizon = start["stop"]

        assert len(cumul) > 0
        result = sum(interval for interval in cumul)  # type: ignore
        return result.sort_values("start")  # type: ignore

    def __or__(
        self, other: Interval | IntervalCollection
    ) -> IntervalCollection:
        if isinstance(other, Interval):
            other = IntervalCollection(other)

        return self.union(other)

    def __ror__(self, other: Interval) -> IntervalCollection:
        return IntervalCollection(other) | self

    def union(self, other: IntervalCollection) -> IntervalCollection:
        """Returns the result of an union of intervals.

        :param other: the second interval or collection of intervals

        .. note::

            The binary operator `|` is equivalent to this method.

        """
        return (self + other).consolidate()

    def __and__(
        self, other: Interval | IntervalCollection
    ) -> None | IntervalCollection:
        return self.intersection(other)

    def __rand__(self, other: Interval) -> None | IntervalCollection:
        return self.intersection(other)

    def intersection(
        self, other: Interval | IntervalCollection
    ) -> None | IntervalCollection:
        """Returns the result of an intersection of intervals.

        :param other: the second interval or collection of intervals

        :return: may be `None` if the intersection is empty

        .. note::

            The binary operator `&` is equivalent to this method.

        """
        compiled = self.consolidate()

        if isinstance(other, IntervalCollection):
            cumul = []
            for segment in other.consolidate():
                result = compiled.intersection(segment)
                if result is not None:
                    for segment in result:
                        cumul.append(segment)
            if len(cumul) == 0:
                return None
            return sum(interval for interval in cumul)  # type: ignore

        if isinstance(other, Interval):
            cumul = []
            for segment in compiled:
                if segment.overlap(other):
                    start = max(segment.start, other.start)
                    stop = min(segment.stop, other.stop)
                    cumul.append(Interval(start, stop))
            if len(cumul) == 0:
                return None
            return sum(interval for interval in cumul)  # type: ignore

        return NotImplemented
