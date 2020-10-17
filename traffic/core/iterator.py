import functools
import operator
from typing import TYPE_CHECKING, Callable, Iterator, Optional, cast

if TYPE_CHECKING:
    from . import Flight  # noqa: F401


class FlightIterator:
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
            for _, elt in enumerate(self):
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
        return self.next() is not None  # noqa: B305

    def next(self) -> Optional["Flight"]:
        return next((segment for segment in self), None)

    def sum(self) -> int:
        return len(self)

    def all(self) -> Optional["Flight"]:
        from traffic.core import Flight  # noqa: F811

        t = sum(flight for flight in self)
        if t == 0:
            return None
        return Flight(t.data)  # type: ignore

    def max(self, key: str = "duration") -> Optional["Flight"]:
        return max(self, key=lambda x: getattr(x, key), default=None)

    def min(self, key: str = "duration") -> Optional["Flight"]:
        return min(self, key=lambda x: getattr(x, key), default=None)

    def plot(self, *args, **kwargs) -> None:
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
