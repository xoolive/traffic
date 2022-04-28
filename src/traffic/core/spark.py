from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, Union, cast

from pyspark.sql import DataFrame as SDF
from pyspark.sql import functions

import numpy as np
import pandas as pd

from .flight import Flight


class Implementation(Protocol):
    __name__: str

    def __call__(self, df: SDF, *args: Any, **kwargs: Any) -> SDF:
        ...


_implementations: dict[str, Implementation] = dict()


def spark_register(func: Implementation) -> Implementation:
    _implementations[func.__name__] = func
    return func


def wraps(name: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
    def udf_wrapper(df: pd.DataFrame) -> pd.DataFrame:
        result = cast(
            Union[None, bool, np.bool_, Flight],
            getattr(Flight, name)(Flight(df), *args, **kwargs),
        )
        if result is False or result is np.False_:
            return df.query("icao24 != icao24")
        if result is True or result is np.True_:
            return df
        # this needs to change to allow for columns to be added
        return result.data[df.columns]  # type: ignore

    return udf_wrapper


def get_implementation(
    name: str, *args: Any, **kwargs: Any
) -> Callable[[SDF], SDF]:

    func = _implementations.get(name, None)

    def spark_wrapper(sdf: SDF) -> SDF:
        if func is None:
            # default implementation: wrap the call in a UDF
            return sdf.groupby("flight_id").applyInPandas(
                wraps(name, *args, **kwargs), schema=sdf.schema
            )
        else:
            # otherwise just apply the provided implementation
            logging.info(f"Found a specific Spark implementation for {name}")
            return func(sdf, *args, **kwargs)

    return spark_wrapper


@spark_register
def query(df: SDF, condition: str) -> SDF:
    return df.filter(condition)


def feature_lt(
    sdf: SDF,
    feature: str | Callable[["Flight"], Any],
    value: Any,
    strict: bool = True,
) -> SDF:

    if isinstance(feature, str):
        *name_split, agg = feature.split("_")
        feature = "_".join(name_split)
        agg_fun = getattr(functions, agg, None)
        if feature in sdf.columns and agg_fun is not None:
            op = "<" if strict else "<="
            flight_ids = (
                sdf.groupby("flight_id")
                .agg(agg_fun(feature).alias("feature"))
                .filter(f"feature {op} {value}")
            )
            return sdf.join(flight_ids, on="flight_id", how="leftsemi")

    logging.info(f"Falling back to default UDF implementation for {feature}")
    return sdf.groupby("flight_id").applyInPandas(
        wraps("feature_lt", feature, value, strict=True), schema=sdf.schema
    )
