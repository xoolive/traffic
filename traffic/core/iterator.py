import functools
import operator
from typing import TYPE_CHECKING, Callable, Iterator, Optional, cast

if TYPE_CHECKING:
    from . import Flight  # noqa: F401
    from . import Traffic  # noqa: F401
    from .lazy import LazyTraffic  # noqa: F401


class FlightIterator:
    """
    A FlightIterator is a specific structure providing helpers after methods
    applied on a Flight that return a sequence of pieces of trajectories.

    Methods returning a FlightIterator include:

    - ``Flight.split("10T")`` iterates over pieces of trajectories separated by
      more than 10 minutes without data;
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

    def __init__(self, generator):
        self.generator = generator

    def __next__(self):
        return next(self.generator)

    def __iter__(self):
        yield from self.generator

    def __len__(self):
        return sum(1 for _ in self)

    def _repr_html_(self):
        try:
            concat = functools.reduce(operator.or_, self)._repr_html_()
        except TypeError:  # reduce() of empty sequence with no initial value
            concat = "Empty sequence"
        return (
            """<div class='alert alert-warning'>
            <b>Warning!</b>
            This iterable structure is neither a Flight nor a Traffic structure.
            Each corresponding segment of the flight (if any) is displayed
            below.
            <br/>
            Possible utilisations include iterating (for loop, next keyword),
            indexing with the bracket notation (slices supported) and most
            native built-ins made for iterable structures.
            </div>"""
            + concat
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            for i, elt in enumerate(self):
                if i == index:
                    return elt
        if isinstance(index, slice):
            if index.step is not None and index.step <= 0:
                raise ValueError("Negative steps are not supported")

            def gen():
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
        return self.next() is not None  # noqa: B305

    def next(self) -> Optional["Flight"]:
        """Returns the first/next element in the FlightIterator.

        Example usage:

        >>> first_attempt = flight.runway_change().next()

        This is equivalent to:

        >>> flight.next("runway_change")
        """
        return next((segment for segment in self), None)

    def sum(self) -> int:
        """Returns the size of the FlightIterator.

        Example usage:

        >>> flight.go_around().sum()
        1

        This is equivalent to:

        >>> flight.sum("go_around")
        """

        return len(self)

    def all(self) -> Optional["Flight"]:
        """Returns the concatenation of elements in the FlightIterator.

        >>> flight.aligned_on_ils("LFBO").all()

        This is equivalent to:

        >>> flight.all(lambda f: f.aligned_on_ils("LFBO"))

        """
        from traffic.core import Flight  # noqa: F811

        t = sum(flight.assign(index_=i) for i, flight in enumerate(self))
        if t == 0:
            return None
        return Flight(t.data)  # type: ignore

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

    def __call__(
        self, fun: Callable[..., "LazyTraffic"], *args, **kwargs,
    ) -> Optional["Flight"]:
        from traffic.core import Flight, Traffic  # noqa: F811

        in_ = Traffic.from_flights(
            segment.assign(index_=i) for i, segment in enumerate(self)
        )
        if in_ is None:
            return None
        out_ = fun(in_, *args, **kwargs).eval()
        if out_ is None:
            return None

        return Flight(out_.data)

    def plot(self, *args, **kwargs) -> None:
        """Plots all elements in the structure.

        Arguments as passed as is to the `Flight.plot()` method.
        """
        for segment in self:
            segment.plot(*args, **kwargs)


def flight_iterator(
    fun: Callable[..., Iterator["Flight"]]
) -> Callable[..., FlightIterator]:

    msg = (
        "The @flight_iterator decorator can only be set on methods "
        ' annotated with an Iterator["Flight"] return type.'
    )
    if fun.__annotations__["return"] != Iterator["Flight"]:
        raise TypeError(msg)

    @functools.wraps(fun, updated=("__dict__", "__annotations__"))
    def fun_wrapper(*args, **kwargs) -> FlightIterator:
        return FlightIterator(fun(*args, **kwargs))

    fun_wrapper = cast(Callable[..., FlightIterator], fun_wrapper)
    fun_wrapper.__annotations__["return"] = FlightIterator

    return fun_wrapper
