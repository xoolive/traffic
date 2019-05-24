import functools
import inspect
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, List, Optional

from tqdm.autonotebook import tqdm

from .flight import Flight
from .mixins import GeographyMixin

if TYPE_CHECKING:
    from .traffic import Traffic  # noqa: F401


class LazyLambda:
    """
    This callable is stacked for future operations on Flights.

    .. warning::
        A non-class implementation is possible with nested functions but the
        result would not be pickable and raise issues during multiprocessing.

    """

    def __init__(
        self, f_name: str, idx_name: Optional[str], *args, **kwargs
    ) -> None:
        self.f_name = f_name
        self.idx_name = idx_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, idx: int, elt: Optional[Flight]) -> Optional[Flight]:
        if self.idx_name is not None:
            self.kwargs[self.idx_name] = idx
        result = getattr(Flight, self.f_name)(elt, *self.args, **self.kwargs)
        if result is False:
            return None
        if result is True:
            return elt
        return result


def apply(
    stacked_ops: List[LazyLambda], idx: int, flight: Flight
) -> Optional["Flight"]:
    """Recursively applies all operations on each Flight.

    - *map* operations return Flight
    - *filter* operations return a Flight if True, otherwise None

    Note that the only valid reduce operation is `Traffic.from_flights`.
    """
    return functools.reduce(  # type: ignore
        (
            lambda next_step, fun: fun(idx, next_step)  # type: ignore
            if next_step is not None
            else None
        ),
        stacked_ops,
        flight,
    )


class LazyTraffic:
    """

    In the following example, ``lazy_t`` is not evaluated:

    >>> lazy_t = t.filter().resample('10s')
    >>> type(t_lazy)
    traffic.core.lazy.LazyTraffic

    """

    def __init__(self, wrapped_t: "Traffic", stacked_ops: List[LazyLambda]):
        self.wrapped_t: "Traffic" = wrapped_t
        self.stacked_ops: List[LazyLambda] = stacked_ops

    def __repr__(self):
        assert LazyTraffic.__doc__ is not None
        return (
            f"{super()}\n"
            + LazyTraffic.__doc__
            + f"Call eval() to apply {len(self.stacked_ops)} stacked operations"
        )

    def eval(
        self, max_workers: int = 1, desc: Optional[str] = None
    ) -> "Traffic":
        """

        The result can only be accessed after a call to ``eval()``.

        max_workers: int, default: 1
            Multiprocessing is usually worth it. However, a sequential
            processing is triggered by default. Keep the value close to the
            number of cores of your processor. If memory becomes a problem,
            stick to the default.

        desc: str, default: None
            If not None, a tqdm progressbar is displayed with this parameter.

        Example usage:
            The following call

            >>> t_lazy.eval(max_workers=4, desc="preprocessing")

            is equivalent to the multiprocessed version of

            >>> Traffic.from_flights(
            ...     flight.filter().resample("10s")
            ...     for flight in tqdm(t, desc="preprocessing")
            ... )

        When many operations are stacked, this call is more efficient, esp. on
        large structures, than as many full iterations on the Traffic structure.

        Backward compatibility is ensured by an automatic call to eval() with
        default options.

        >>> t_lazy.to_pickle("output_file.pkl")
        WARNING:root:.eval() has been automatically appended for you.
        Check the documentation for more options.

        """

        if max_workers < 2:
            iterator = self.wrapped_t
            if desc is not None:
                iterator = tqdm(
                    iterator, total=len(self.wrapped_t), desc=desc, leave=False
                )
            cumul = list(
                apply(self.stacked_ops, idx, flight)
                for idx, flight in enumerate(iterator)
            )
        else:
            cumul = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = {
                    executor.submit(
                        apply, self.stacked_ops, idx, flight
                    ): flight
                    for idx, flight in enumerate(self.wrapped_t)
                }
                tasks_completed = as_completed(tasks)
                if desc is not None:
                    tasks_completed = tqdm(
                        tasks_completed,
                        total=len(tasks),
                        desc=desc,
                        leave=False,
                    )
                for future in tasks_completed:
                    cumul.append(future.result())

        # return Traffic.from_flights
        return self.wrapped_t.__class__.from_flights(
            [flight for flight in cumul if flight is not None]
        )

    def __getattr__(self, name):
        if hasattr(self.wrapped_t, name):
            logging.warn(
                ".eval() has been automatically appended for you.\n"
                "Check the documentation for more options."
            )
            return getattr(self.eval(), name)


def lazy_evaluation(
    idx_name: Optional[str] = None, default: Optional[bool] = False
) -> Callable[..., LazyTraffic]:
    """A decorator to delegate methods to Flight in a lazy manner.

    Each decorated Traffic method returns a LazyTraffic structure with the
    corresponding operation stacked.

    When the `default` option is set to True, the method returns a Traffic when
    called on Traffic but is stacked before applications on Flight objects if
    called on a LazyTraffic.

    Parameters
    ----------

    idx_name: str, None, default: None
        If the method needs the index (from `enumerate`) produced during the
        iteration, specify the name of the corresponding argument.
        (see {Traffic, Flight}.assign_id for an example)
    default: bool, default: False
        If set to True, the Traffic implementation is used when called on a
        Traffic structure; and the Flight implementation is stacked when called
        on a LazyTraffic structure.

    Returns
    -------
        A decorated Traffic method

    """

    def wrapper(f):

        # Check the decorated method is implemented by A
        if not hasattr(Flight, f.__name__):
            raise TypeError(f"Class Flight does not provide {f.__name__}")

        def lazy_λf(lazy: LazyTraffic, *args, **kwargs):
            op_idx = LazyLambda(f.__name__, idx_name, *args, **kwargs)
            return LazyTraffic(lazy.wrapped_t, lazy.stacked_ops + [op_idx])

        lazy_λf.__annotations__ = getattr(Flight, f.__name__).__annotations__
        lazy_λf.__annotations__["self"] = LazyTraffic
        lazy_λf.__annotations__["return"] = LazyTraffic

        # Attach the method to LazyCollection for further chaining
        setattr(LazyTraffic, f.__name__, lazy_λf)

        if default is True:
            if f.__doc__ is not None:
                f.__doc__ += f"""\n        .. note::
            This method will use the Flight `implementation
            <traffic.core.flight.html#traffic.core.Flight.{f.__name__}>`_ when
            stacked for lazy evaluation.  """
            return f

        # Take the method in Flight and create a LazyCollection
        def λf(wrapped_t: "Traffic", *args, **kwargs):
            op_idx = LazyLambda(f.__name__, idx_name, *args, **kwargs)
            return LazyTraffic(wrapped_t, [op_idx])

        if f.__doc__ is not None:
            λf.__doc__ = f.__doc__
        else:
            λf.__doc__ = getattr(Flight, f.__name__).__doc__

        λf.__annotations__ = getattr(Flight, f.__name__).__annotations__

        if λf.__doc__ is not None:
            λf.__doc__ += """\n        .. warning::
            This method will be stacked for lazy evaluation.  """

        return λf

    return wrapper


# All methods coming from DataFrameMixin and GeographyMixin make sense in both
# Flight and Traffic. However it would give real hard headaches to decorate them
# all *properly* in the Traffic implementation. The following monkey-patching
# does it all based on the type annotations (must match (T, ...) -> T)


for name, handle in inspect.getmembers(
    GeographyMixin, predicate=inspect.isfunction
):
    annots = handle.__annotations__
    if name.startswith("_") or "self" not in annots or "return" not in annots:
        continue

    if annots["self"] is annots["return"]:

        def make_lambda(name: str) -> Callable[..., LazyTraffic]:
            def lazy_λf(lazy: LazyTraffic, *args, **kwargs):
                op_idx = LazyLambda(name, None, *args, **kwargs)
                return LazyTraffic(lazy.wrapped_t, lazy.stacked_ops + [op_idx])

            lazy_λf.__doc__ = handle.__doc__
            lazy_λf.__annotations__ = handle.__annotations__

            return lazy_λf

        # Attach the method to LazyCollection for further chaining
        setattr(LazyTraffic, name, make_lambda(name))
