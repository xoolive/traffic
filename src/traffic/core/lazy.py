from __future__ import annotations

import ast
import functools
import inspect
import logging
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from tokenize import OP, generate_tokens, untokenize
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Dict,
    List,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    cast,
    overload,
)

import astor

import numpy as np
import pandas as pd

from . import tqdm
from .flight import Flight
from .mixins import GeographyMixin

if TYPE_CHECKING:
    from .traffic import Traffic

_log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

TrafficMethod = Callable[Concatenate["Traffic", P], R]

# TODO PEP 563 (Postponed Evaluation of Annotations) from Python 3.11

# This syntax from Python 3.12
# type TrafficMethod[P] =
#  Callable[Concatenate[Traffic, P], bool | None | Traffic]


class FaultCatcher:
    flag: bool = False
    flight: Optional[Flight] = None

    def __enter__(self) -> None:
        FaultCatcher.flag = True
        FaultCatcher.flight = None

    def __exit__(
        self, _exc_type: Any, _exc_value: Any, tb: Optional[types.TracebackType]
    ) -> None:
        FaultCatcher.flag = False
        traceback.format_tb(tb)


class LambdaParseException(RuntimeError):
    pass


class LazyLambda:
    """
    This callable is stacked for future operations on Flights.

    .. warning::
        A non-class implementation is possible with nested functions but the
        result would not be pickable and raise issues during multiprocessing.

    """

    def __init__(
        self, f_name: str, idx_name: Optional[str], *args: Any, **kwargs: Any
    ) -> None:
        self.f_name = f_name
        self.idx_name = idx_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, idx: int, elt: Optional[Flight]) -> Optional[Flight]:
        if self.idx_name is not None:
            self.kwargs[self.idx_name] = idx
        result = cast(
            Union[None, bool, np.bool_, Flight],
            getattr(Flight, self.f_name)(elt, *self.args, **self.kwargs),
        )
        if result is False or result is np.False_:
            return None
        if result is True or result is np.True_:
            return elt
        assert not isinstance(result, np.bool_)
        return result


def apply(
    stacked_ops: List[LazyLambda], idx: int, flight: Optional[Flight]
) -> Optional["Flight"]:
    """Recursively applies all operations on each Flight.

    - *map* operations return Flight
    - *filter* operations return a Flight if True, otherwise None

    Note that the only valid reduce operation is `Traffic.from_flights`.
    """
    return functools.reduce(
        (
            lambda next_step, fun: fun(idx, next_step)
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

    You need to call the ``.eval()`` method for that.

    """

    def __init__(
        self,
        wrapped_t: "Traffic",
        stacked_ops: List[LazyLambda],
        iterate_kw: Optional[Dict[str, Any]] = None,
        tqdm_kw: Optional[Dict[str, Any]] = None,
    ):
        self.wrapped_t: "Traffic" = wrapped_t
        self.stacked_ops: List[LazyLambda] = stacked_ops
        self.iterate_kw: Dict[str, Any] = (
            iterate_kw if iterate_kw is not None else {}
        )
        self.tqdm_kw: Dict[str, Any] = tqdm_kw if tqdm_kw is not None else {}

    def __repr__(self) -> str:
        assert LazyTraffic.__doc__ is not None
        return (
            "class LazyTraffic:\n"
            + LazyTraffic.__doc__
            + "\n"
            + f"Call eval() to apply {len(self.stacked_ops)} stacked operations"
        )

    def eval(
        self,
        max_workers: int = 1,
        desc: None | str = None,
        cache_file: str | Path | None = None,
    ) -> None | "Traffic":
        """

        The result can only be accessed after a call to ``eval()``.

        :param max_workers: (default: 1)
            Multiprocessing is usually worth it. However, a sequential
            processing is triggered by default. Keep the value close to the
            number of cores of your processor. If memory becomes a problem,
            stick to the default.

        :param desc: (default: None)
            If not None, a tqdm progressbar is displayed with this parameter.

        :param cache_file: (default: None)
            If not None, store the results in cache_file and load the results
            from the file if it exists.

        **Example usage:**

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

        if cache_file is not None and Path(cache_file).exists():
            return self.wrapped_t.__class__.from_file(cache_file)

        if max_workers < 2 or FaultCatcher.flag is True:
            iterator = self.wrapped_t.iterate(**self.iterate_kw)
            # not the same iterator to not exhaust it
            total = sum(1 for _ in self.wrapped_t.iterate(**self.iterate_kw))
            if desc is not None or len(self.tqdm_kw) > 0:
                tqdm_kw = {
                    **dict(desc=desc, leave=False, total=total),
                    **self.tqdm_kw,
                }
                iterator = tqdm(iterator, **tqdm_kw)

            if FaultCatcher.flag is True:
                try:
                    cumul = list()
                    for idx, flight in enumerate(iterator):
                        cumul.append(apply(self.stacked_ops, idx, flight))
                except Exception as e:
                    FaultCatcher.flight = flight
                    raise e

            else:
                cumul = list(
                    apply(self.stacked_ops, idx, flight)
                    for idx, flight in enumerate(iterator)
                )
        else:
            cumul = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                iterator = self.wrapped_t.iterate(**self.iterate_kw)
                if len(self.tqdm_kw):
                    iterator = tqdm(iterator, **self.tqdm_kw)
                tasks = {
                    executor.submit(
                        apply, self.stacked_ops, idx, flight
                    ): flight
                    for idx, flight in enumerate(iterator)
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
        if len(cumul) == 0 or all(elt is None for elt in cumul):
            result = None
        elif any(isinstance(elt, Flight) for elt in cumul):
            result = self.wrapped_t.__class__.from_flights(
                [flight for flight in cumul if flight is not None]
            )
        elif any(isinstance(elt, dict) for elt in cumul):
            result = pd.DataFrame.from_records(
                [elt for elt in cumul if elt is not None]
            )
        else:
            result = pd.concat(cumul)

        if cache_file is not None and result is not None:
            if Path(cache_file).suffix == ".parquet":
                result.to_parquet(cache_file)
            else:
                result.to_pickle(cache_file)

        return result

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.wrapped_t, name):
            _log.warning(
                ".eval() has been automatically appended for you.\n"
                "Check the documentation for more options."
            )
            return getattr(self.eval(), name)
        raise NotImplementedError(
            f"Method '{name}' not implemented on Traffic or LazyTraffic"
        )


@overload
def lazy_evaluation(
    default: "Literal[None, False]" = False,
    idx_name: Optional[str] = None,
) -> Callable[
    [TrafficMethod[P, R]], Callable[Concatenate["Traffic", P], LazyTraffic]
]: ...


@overload
def lazy_evaluation(
    default: "Literal[True]",
    idx_name: Optional[str] = None,
) -> Callable[[TrafficMethod[P, R]], TrafficMethod[P, R]]: ...


def lazy_evaluation(
    default: None | bool = False,
    idx_name: None | str = None,
) -> Callable[
    [TrafficMethod[P, R]],
    Callable[Concatenate["Traffic", P], R | LazyTraffic],
]:
    """A decorator to delegate methods to :class:`~traffic.core.Flight` in a
    lazy manner.

    Each decorated :class:`~traffic.core.Traffic` method returns a
    :class:`~traffic.core.LazyTraffic` structure with the corresponding
    operation stacked.

    When the `default` option is set to True, the method returns a
    :class:`~traffic.core.Traffic` when called on :class:`~traffic.core.Traffic`
    but is stacked before applications on :class:`~traffic.core.Flight` objects
    if called on a :class:`~traffic.core.LazyTraffic`.

    :param default: (default: False)
        If set to True, the :class:`~traffic.core.Traffic` implementation is
        used when called on a :class:`~traffic.core.Traffic` structure; and the
        :class:`~traffic.core.Flight` implementation is stacked when called on a
        :class:`~traffic.core.LazyTraffic` structure.

    :param idx_name: (default: None)
        If the method needs the index (from `enumerate`) produced during the
        iteration, specify the name of the corresponding argument.  (see
        {:class:`~traffic.core.Traffic`,
        :class:`~traffic.core.Flight`}.assign_id for an example)

    """

    def wrapper(
        f: TrafficMethod[P, R],
    ) -> Callable[Concatenate["Traffic", P], R | LazyTraffic]:
        # Check parameters passed (esp. filter_if) are not lambda because those
        # are not serializable therefore **silently** fail when multiprocessed.
        msg = """
{method}(lambda f: ...) will *silently* fail when evaluated on several cores.
You may write {method}("lambda f: ..."); traffic tries to do it for you but the
feature is experimental.
        """

        def is_lambda(f: Any) -> bool:
            return isinstance(f, types.LambdaType) and f.__name__ == "<lambda>"

        # Check the decorated method is implemented by A
        if not hasattr(Flight, f.__name__):
            raise TypeError(f"Class Flight does not provide {f.__name__}")

        def stringify_lambda(lambda_func: Callable[..., Any]) -> str:
            src_list, *_ = inspect.getsourcelines(lambda_func)
            # Give up if the lambda is more than one line
            if len(src_list) > 1:
                raise LambdaParseException(msg)
            src1 = src_list[0].strip()
            # sometimes the lines starts with a dot and is not parsable
            tokens = [t[:2] for t in generate_tokens(StringIO(src1).readline)]
            if tokens[0][0] == OP:
                tokens.pop(0)
            lambda_line_src: str = untokenize(tokens)
            ast_ = ast.parse(lambda_line_src)
            ast_lambda = next(
                (
                    node
                    for node in ast.walk(ast_)
                    if isinstance(node, ast.Lambda)
                ),
                None,
            )
            if ast_lambda is None:
                raise LambdaParseException(msg)

            return cast(str, astor.to_source(ast_lambda))

        def lazy_λf(
            self: LazyTraffic, /, *args: P.args, **kwargs: P.kwargs
        ) -> LazyTraffic:
            if any(is_lambda(arg) for arg in args):
                _log.warning(msg.format(method=f.__name__))
                args = tuple(  # type: ignore
                    stringify_lambda(arg)  # type: ignore
                    if not isinstance(arg, str) and is_lambda(arg)
                    else arg
                    for arg in args
                )
            if any(is_lambda(arg) for arg in kwargs.values()):
                _log.warning(msg.format(method=f.__name__))
                kwargs = dict(  # type: ignore
                    (
                        key,
                        stringify_lambda(arg)  # type: ignore
                        if not isinstance(arg, str) and is_lambda(arg)
                        else arg,
                    )
                    for key, arg in kwargs.items()
                )

            op_idx = LazyLambda(f.__name__, idx_name, *args, **kwargs)

            return LazyTraffic(
                self.wrapped_t,
                [*self.stacked_ops, op_idx],
                self.iterate_kw,
                self.tqdm_kw,
            )

        lazy_λf.__annotations__ = dict(  # make a copy!!
            getattr(Flight, f.__name__).__annotations__
        )
        lazy_λf.__annotations__["return"] = LazyTraffic

        # Attach the method to LazyCollection for further chaining
        setattr(LazyTraffic, f.__name__, lazy_λf)

        if default is True:
            if f.__doc__ is not None:
                f.__doc__ += """
        .. note::

            This method will use the :class:Flight implementation
            when stacked for lazy evaluation.
                """
            return f

        # Take the method in Flight and create a LazyCollection
        def λf(
            wrapped_t: "Traffic",
            /,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> LazyTraffic:
            op_idx = LazyLambda(f.__name__, idx_name, *args, **kwargs)

            if any(is_lambda(arg) for arg in args):
                _log.warning(msg.format(method=f.__name__))
            if any(is_lambda(arg) for arg in kwargs.values()):
                _log.warning(msg.format(method=f.__name__))

            return LazyTraffic(wrapped_t, [op_idx])

        if f.__doc__ is not None:
            λf.__doc__ = f.__doc__
        else:
            λf.__doc__ = getattr(Flight, f.__name__).__doc__

        λf.__annotations__ = dict(  # make a copy!!
            getattr(Flight, f.__name__).__annotations__
        )
        λf.__annotations__["return"] = LazyTraffic

        if λf.__doc__ is not None:
            λf.__doc__ += """

        .. warning::

            This method will be stacked for lazy evaluation.  """

        return λf

    return wrapper


# All methods coming from DataFrameMixin and GeographyMixin make sense in both
# Flight and Traffic. However it would give real hard headaches to decorate them
# all *properly* in the Traffic implementation. The following monkey-patching
# does it all based on the type annotations (must match (T, ...) -> T or
# (T, ...) -> Optional[T])

for name, handle in inspect.getmembers(
    GeographyMixin, predicate=inspect.isfunction
):
    annots = handle.__annotations__
    if name.startswith("_") or "return" not in annots:
        continue

    # includes .query(), etc.
    if (
        annots["return"] == "Self"
        or annots["return"] == "Optional[Self]"
        or annots["return"] == "None | Self"
    ):

        def make_lambda(
            handle: Callable[Concatenate[GeographyMixin, P], Any],
        ) -> Callable[Concatenate[LazyTraffic, P], LazyTraffic]:
            def lazy_λf(
                self: LazyTraffic, /, *args: P.args, **kwargs: P.kwargs
            ) -> LazyTraffic:
                op_idx = LazyLambda(handle.__name__, None, *args, **kwargs)
                return LazyTraffic(
                    self.wrapped_t,
                    [*self.stacked_ops, op_idx],
                    self.iterate_kw,
                    self.tqdm_kw,
                )

            lazy_λf.__name__ = handle.__name__
            lazy_λf.__qualname__ = f"LazyTraffic.{handle.__name__}"
            lazy_λf.__doc__ = handle.__doc__
            lazy_λf.__annotations__ = handle.__annotations__.copy()
            lazy_λf.__annotations__["return"] = "Self"

            return lazy_λf

        # Attach the method to LazyCollection for further chaining
        setattr(LazyTraffic, name, make_lambda(handle))
