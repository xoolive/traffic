import json
from collections import UserDict
from hashlib import md5
from inspect import currentframe, signature
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import pandas as pd

if TYPE_CHECKING:
    from . import Flight, Traffic


class property_cache(object):
    """Computes attribute value and caches it in the instance.
    Python Cookbook (Denis Otkidach)
    https://stackoverflow.com/users/168352/denis-otkidach

    This decorator allows you to create a property which can be computed once
    and accessed many times. Sort of like memoization, but by instance.

    """

    def __init__(self, method):
        # record the unbound-method and the name
        self.method = method
        self.name = method.__name__
        self.__doc__ = method.__doc__
        self.__annotations__ = method.__annotations__

    def __get__(self, instance, cls):
        # self: <__main__.cache object at 0xb781340c>
        # inst: <__main__.A object at 0xb781348c>
        # cls: <class '__main__.A'>
        if instance is None:
            # instance attribute accessed on class, return self
            # You get here if you write `A.bar`
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(instance)
        # setattr redefines the instance's attribute
        setattr(instance, self.name, result)
        return result


class Cache(UserDict):
    def __init__(self, cachedir: Path) -> None:
        self.cachedir = cachedir
        if not self.cachedir.exists():
            self.cachedir.mkdir(parents=True)
        super().__init__()

    def __missing__(self, hashcode: str):
        filename = self.cachedir / f"{hashcode}.json"
        if filename.exists():
            response = json.loads(filename.read_text())
            # Do not overwrite the file!
            super().__setitem__(hashcode, response)
            return response

    def __setitem__(self, hashcode: str, data):
        super().__setitem__(hashcode, data)
        filename = self.cachedir / f"{hashcode}.json"
        with filename.open("w") as fh:
            fh.write(json.dumps(data))


T = TypeVar("T", pd.DataFrame, "Traffic", "Flight")


def cache_results(
    fun: Optional[Callable[..., T]] = None,
    cache_directory: Path = Path("."),
    loader: Callable[[Path], T] = pd.read_pickle,
    pd_varnames: bool = False,
):
    """
    The point of this method is to be able to cache results of some costly
    functions on pd.DataFrame, Flight or Traffic structures.

    Decorate your function with the cache_results method and go ahead!


    """

    def cached_values(fun: Callable[..., T]) -> Callable[..., T]:
        def newfun(*args, **kwargs) -> T:
            global callers_local_vars
            sig = signature(fun)

            if sig.return_annotation is not pd.DataFrame:
                raise TypeError(
                    "The wrapped function must have a return type of "
                    "pandas DataFrame and be annotated as so."
                )

            bound_args = sig.bind(*args, **kwargs)
            all_args = {
                **dict(
                    (param.name, param.default)
                    for param in sig.parameters.values()
                ),
                **dict(bound_args.arguments.items()),
            }

            l_vars = currentframe().f_back.f_locals.items()  # type: ignore

            args_ = list()
            for value in all_args.values():
                if isinstance(value, pd.DataFrame) or (
                    hasattr(value, "data")
                    and isinstance(value.data, pd.DataFrame)
                ):
                    attempt = None
                    if pd_varnames:
                        attempt = next(
                            (
                                var_name
                                for var_name, var_val in l_vars
                                if var_val is value
                            ),
                            None,
                        )

                    if attempt is not None:
                        args_.append(attempt)
                    else:
                        args_.append(md5(value.values.tobytes()).hexdigest())
                else:
                    args_.append(f"{value}")

            filepath = cache_directory / (
                fun.__name__ + "_" + "_".join(args_) + ".pkl"
            )

            if filepath.exists():
                print(f"Reading cached data from {filepath}")
                return loader(filepath)

            res = fun(*args, **kwargs)
            res.to_pickle(filepath)
            return res

        return newfun

    if fun is None:
        return cached_values
    else:
        return cached_values(fun)
