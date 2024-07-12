from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Union

import rich.repr

if TYPE_CHECKING:
    from . import Flight
    from .lazy import LazyTraffic
    from .mixins import _HBox


@rich.repr.auto()
class FlightIterator:
    """
    A FlightIterator is a specific structure providing helpers after methods
    applied on a Flight that return a sequence of pieces of trajectories.

    Methods returning a FlightIterator include:

    - ``Flight.split("10 min")`` iterates over pieces of trajectories separated
      by more than 10 minutes without data;
    - ``Flight.go_around("LFBO")`` iterates over landing attempts on a given
      airport;
    - ``Flight.aligned_on_ils("LFBO")`` iterates over segments of trajectories
      aligned with any of the runways at LFBO.
    - and more.

    Since a FlightIterator is not a Flight, you can:

    - iterate on it with a for loop, or with Python built-ins functions;
    - index it with bracket notation (using positive integers or slices);
    - get True if the sequence is non empty with ``.has()``;
    - get the first element in the sequence with ``.next()``;
    - count the element in the sequence with ``.sum()``;
    - concatenate all elements in the sequence with ``.all()``;
    - get the biggest/shortest element with ``.max()``/``.min()``. By default,
      comparison is made on duration.

    .. warning::

        **FlightIterator instances consume themselves out**.

        If you store a FlightIterator in a variable, calling methods twice in a
        row will yield different results. In Jupyter environments, representing
        the FlightIterator will consume it too.

        To avoid issues, the best practice is to **not** store any
        FlightIterator in a variable.

    """

    def __init__(self, generator: Iterator["Flight"]) -> None:
        self.generator = generator
        self.cache: list["Flight"] = list()
        self.iterator: None | Iterator["Flight"] = None

    def __next__(self) -> "Flight":
        if self.iterator is None:
            self.iterator = iter(self)
        return next(self.iterator)

    def __iter__(self) -> Iterator["Flight"]:
        yield from self.cache
        for elt in self.generator:
            self.cache.append(elt)
            yield elt

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @functools.lru_cache()
    def _repr_html_(self) -> str:
        title = "<h3><b>FlightIterator</b></h3>"
        concat: None | "Flight" | "_HBox" = None
        for segment in self:
            concat = segment if concat is None else concat | segment
        return title + (
            concat._repr_html_() if concat is not None else "Empty sequence"
        )

    def __rich_repr__(self) -> rich.repr.Result:
        for i, segment in enumerate(self):
            if i == 0:
                if segment.flight_id:
                    yield segment.flight_id
                else:
                    yield "icao24", segment.icao24
                    yield "callsign", segment.callsign
                    yield "start", format(segment.start)
            yield f"duration_{i}", format(segment.duration)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union["Flight", "FlightIterator"]:
        if isinstance(index, int):
            for i, elt in enumerate(self):
                if i == index:
                    return elt
        if isinstance(index, slice):
            if index.step is not None and index.step <= 0:
                raise ValueError("Negative steps are not supported")

            def gen() -> Iterator["Flight"]:
                assert isinstance(index, slice)
                modulo_start = None
                for i, elt in enumerate(self):
                    if index.start is not None and i < index.start:
                        continue
                    if index.stop is not None and i >= index.stop:
                        continue

                    if modulo_start is None:
                        base = index.step if index.step is not None else 1
                        modulo_start = i % base

                    if i % base == modulo_start:
                        yield elt

            return self.__class__(gen())
        raise TypeError("The index must be an integer or a slice")

    def has(self) -> bool:
        """Returns True if the FlightIterator is not empty.

        Example usage:

        >>> flight.emergency().has()
        True

        This is equivalent to:

        >>> flight.has("emergency")
        """
        return self.next() is not None

    def next(self) -> Optional["Flight"]:
        """Returns the first/next element in the FlightIterator.

        Example usage:

        >>> first_attempt = flight.runway_change().next()

        This is equivalent to:

        >>> flight.next("runway_change")
        """
        return next(self, None)

    def final(self) -> Optional["Flight"]:
        """Returns the final (last) element in the FlightIterator.

        Example usage:

        >>> first_attempt = flight.runway_change().final()

        This is equivalent to:

        >>> flight.final("runway_change")
        """
        segment = None
        for segment in self:
            continue
        return segment

    def sum(self) -> int:
        """Returns the size of the FlightIterator.

        Example usage:

        >>> flight.go_around().sum()
        1

        This is equivalent to:

        >>> flight.sum("go_around")
        """

        return len(self)

    def all(self, flight_id: None | str = None) -> Optional["Flight"]:
        """Returns the concatenation of elements in the FlightIterator.

        >>> flight.aligned_on_ils("LFBO").all()

        This is equivalent to:

        >>> flight.all(lambda f: f.aligned_on_ils("LFBO"))
        >>> flight.all('aligned_on_ils("LFBO")')

        """
        from traffic.core import Flight, Traffic

        if flight_id is None:
            t = Traffic.from_flights(flight for i, flight in enumerate(self))
        else:
            t = Traffic.from_flights(
                flight.assign(flight_id=flight_id.format(self=flight, i=i))
                for i, flight in enumerate(self)
            )

        if t is None:
            return None

        return Flight(t.data)

    def max(self, key: str = "duration") -> Optional["Flight"]:
        """Returns the biggest element in the Iterator.

        By default, comparison is based on duration.

        >>> flight.query("altitude < 5000").split().max()

        but it can be set on start time as well (the last event to start)

        >>> flight.query("altitude < 5000").split().max(key="start")
        """

        return max(self, key=lambda x: getattr(x, key), default=None)

    def min(self, key: str = "duration") -> Optional["Flight"]:
        """Returns the shortest element in the Iterator.

        By default, comparison is based on duration.

        >>> flight.query("altitude < 5000").split().min()

        but it can be set on ending time as well (the first event to stop)

        >>> flight.query("altitude < 5000").split().min(key="stop")
        """
        return min(self, key=lambda x: getattr(x, key), default=None)

    def map(
        self, fun: Callable[["Flight"], Optional["Flight"]]
    ) -> "FlightIterator":
        """Applies a function on each segment of an Iterator.

        For instance:

        >>> flight.split("10min").map(lambda f: f.resample("2s")).all()

        """

        def aux(self: FlightIterator) -> Iterator["Flight"]:
            for segment in self:
                if (result := fun(segment)) is not None:
                    yield result

        return flight_iterator(aux)(self)

    def __call__(
        self,
        fun: Callable[..., "LazyTraffic"],
        *args: Any,
        **kwargs: Any,
    ) -> Optional["Flight"]:
        from traffic.core import Flight, Traffic

        in_ = Traffic.from_flights(
            segment.assign(index_=i) for i, segment in enumerate(self)
        )
        if in_ is None:
            return None
        out_ = fun(in_, *args, **kwargs).eval()
        if out_ is None:
            return None

        return Flight(out_.data)

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Plots all elements in the structure.

        Arguments as passed as is to the `Flight.plot()` method.
        """
        for segment in self:
            segment.plot(*args, **kwargs)


def flight_iterator(
    fun: Callable[..., Iterator["Flight"]],
) -> Callable[..., FlightIterator]:
    msg = (
        "The @flight_iterator decorator can only be set on methods "
        ' annotated with an Iterator["Flight"] return type.'
        f' Got {fun.__annotations__["return"]}'
    )
    if not (
        fun.__annotations__["return"] == Iterator["Flight"]
        or eval(fun.__annotations__["return"]) == Iterator["Flight"]
    ):
        raise TypeError(msg)

    @functools.wraps(fun, updated=("__dict__", "__annotations__"))
    def fun_wrapper(*args: Any, **kwargs: Any) -> FlightIterator:
        return FlightIterator(fun(*args, **kwargs))

    fun_wrapper.__annotations__["return"] = FlightIterator

    return fun_wrapper
