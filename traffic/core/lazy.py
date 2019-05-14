import functools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, List, Optional

from tqdm.autonotebook import tqdm

from .flight import Flight

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
    """A LazyTraffic wraps a Traffic and operations to apply to each Flight.

    This structure is convenient for chaining.
    Operations are stacked until eval() is called. (see LazyTraffic.eval())

    >>> t = Traffic.from_file("input_file.pkl")
    >>> t_lazy = t.resample('1s').filter()
    >>> type(t_lazy)
    traffic.core.lazy.LazyTraffic

    The underlying Traffic can be accessed after a call to eval().
    If max_workers is greater than 1, automatic multiprocessing is triggered.
    >>> t_processed = t_lazy.eval(max_workers=4, desc="processing")

    Backward compatibility is ensured by an automatic call to eval() with
    default options.

    >>> t_clean.to_pickle("output_file.pkl")
    WARNING:root:.eval() has been automatically appended for you.
    Check the documentation for more options.

    """

    def __init__(self, wrapped_t: "Traffic", stacked_ops: List[LazyLambda]):
        self.wrapped_t: "Traffic" = wrapped_t
        self.stacked_ops: List[LazyLambda] = stacked_ops

    def __repr__(self):
        return (
            f"{super().__repr__()}\n"
            + LazyTraffic.__doc__
            + f"Call eval() to apply {len(self.stacked_ops)} stacked operations"
        )

    def query(self, query_str: str) -> "LazyTraffic":
        """A DataFrame query to apply on each Flight."""
        # Traffic.query could not be properly decorated and typed
        op_idx = LazyLambda("query", None, query_str=query_str)
        return LazyTraffic(self.wrapped_t, self.stacked_ops + [op_idx])

    def eval(
        self, max_workers: int = 1, desc: Optional[str] = None
    ) -> "Traffic":
        """Evaluate all stacked operations.

        Parameters
        ----------

        max_workers: int, default: 1
            The number of processes to launch for evaluating all operations. By
            default, a regular processing is triggered. Ideally this should be
            set to a value close to the number of cores of your processor.
        desc: str, None, default: None
            If not None, a tqdm progressbar is displayed.

        Returns
        -------
        Traffic
            The fully processed Traffic structure.

        Examples
        --------
        >>> t_processed = t_lazy.eval(
        ...     max_workers=4, desc="processing"
        ... )  # doctest: +SKIP

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

        def lazy_lambda_f(lazy: LazyTraffic, *args, **kwargs):
            op_idx = LazyLambda(f.__name__, idx_name, *args, **kwargs)
            return LazyTraffic(lazy.wrapped_t, lazy.stacked_ops + [op_idx])

        # Attach the method to LazyCollection for further chaining
        setattr(LazyTraffic, f.__name__, lazy_lambda_f)

        if default is True:
            if f.__doc__ is not None:
                f.__doc__ += f"""\n        .. note::
            This method will use the Flight `implementation
            <traffic.core.flight.html#traffic.core.Flight.{f.__name__}>`_ when
            stacked for lazy iteration and evaluation.  """
            return f

        # Take the method in Flight and create a LazyCollection
        def lambda_f(wrapped_t: "Traffic", *args, **kwargs):
            op_idx = LazyLambda(f.__name__, idx_name, *args, **kwargs)
            return LazyTraffic(wrapped_t, [op_idx])

        if f.__doc__ is not None:
            lambda_f.__doc__ = f.__doc__
        else:
            lambda_f.__doc__ = getattr(Flight, f.__name__).__doc__

        if lambda_f.__doc__ is not None:
            lambda_f.__doc__ += """\n        .. warning::
            This method will be stacked for lazy iteration and evaluation.  """

        return lambda_f

    return wrapper
