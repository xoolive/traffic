from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, Union, cast, List, Optional

from pyspark.sql import DataFrame as SDF
from pyspark.sql import functions as fn
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

import numpy as np
import pandas as pd

from .flight import Flight


class Implementation(Protocol):
    __name__: str

    def __call__(self, df: SDF, *args: Any, **kwargs: Any) -> SDF:
        ...


_implementations: dict[str, Implementation] = dict()

# column name for groupby
flight_id = "flight_id"


class SparkTraffic:
    def __init__(
        self,
        sdf: SDF,
        cache_after_loading: bool = True,
        assign_flt_id: bool = True,
    ):
        if assign_flt_id:
            sdf = assign_id(sdf)

        if cache_after_loading:
            sdf.cache()

        self.sdf = sdf

    @classmethod
    def from_parquet(
        cls,
        filename: str,
        spark: SparkSession,
        cache_after_loading: bool = True,
        assign_flt_id: bool = True,
    ):
        sdf = spark.read.parquet(filename)
        return cls(sdf, cache_after_loading, assign_flt_id)

    @classmethod
    def from_csv(
        cls,
        filename: str,
        spark: SparkSession,
        cache_after_loading: bool = True,
        assign_flt_id: bool = True,
        time_cols_to_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        sdf = spark.read.csv(filename, **kwargs)

        # convert timestamps when loading from CSV
        if time_cols_to_convert is None:
            # if no columns are given, use default columns with time
            time_cols_to_convert = ["timestamp", "last_position"]
        schema = sdf.schema
        for name, datatype in zip(schema.fieldNames(), schema.fields):
            if name in time_cols_to_convert and datatype != TimestampType:
                sdf = get_timestamp_from_string(sdf, name)

        return cls(sdf, cache_after_loading, assign_flt_id)

    def __getattr__(self, name):
        def func(*args, **kwargs) -> "SparkTraffic":
            callable = getattr(SDF, name, None)
            if callable is None:
                callable = get_implementation(name, *args, **kwargs)
                return SparkTraffic(
                    callable(self.sdf),
                    cache_after_loading=False,
                    assign_flt_id=False,
                )
            # return callable(self.sdf, *args, **kwargs)
            return SparkTraffic(
                callable(
                    self.sdf,
                    *args,
                    **kwargs,
                ),
                cache_after_loading=False,
                assign_flt_id=False,
            )

        return func

    @property
    def traffic(self):
        from .traffic import Traffic

        return self.sdf.toPandas().pipe(Traffic)


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
            return df.iloc[:0, :].copy()
        if result is True or result is np.True_:
            return df
        # this needs to change to allow for columns to be added
        return result.data[df.columns]  # type: ignore

    return udf_wrapper


def get_implementation(
    name: str, *args: Any, **kwargs: Any
) -> Callable[[SDF], SDF]:

    func = _implementations.get(name, None)
    logging.debug(f"get_implementation({name}): {func}")

    def spark_wrapper(sdf: SDF) -> SDF:
        if func is None:
            # default implementation: wrap the call in a UDF
            return sdf.groupby(flight_id).applyInPandas(
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


@spark_register
def feature_lt(
    sdf: SDF,
    feature: str | Callable[["Flight"], Any],
    value: Any,
    strict: bool = True,
) -> SDF:

    if isinstance(feature, str):
        *name_split, agg = feature.split("_")
        feature = "_".join(name_split)
        agg_fun = getattr(fn, agg, None)
        if feature in sdf.columns and agg_fun is not None:
            op = "<" if strict else "<="
            flight_ids = (
                sdf.groupby(flight_id)
                .agg(agg_fun(feature).alias("feature"))
                .filter(f"feature {op} {value}")
            )
            return sdf.join(flight_ids, on=flight_id, how="leftsemi")

    logging.info(f"Falling back to default UDF implementation for {feature}")
    return sdf.groupby(flight_id).applyInPandas(
        wraps("feature_lt", feature, value, strict=True), schema=sdf.schema
    )


# convenient Spark functions, maybe they should be moved
def get_timestamp_from_string(sdf: SDF, col_name: str) -> SDF:
    return sdf.withColumn(
        col_name, fn.to_utc_timestamp(fn.col(col_name), tz="UTC")
    )


def assign_id(
    sdf: SDF,
    identifier: str = "icao24",
    time: str = "timestamp",
    threshold: float = 10 * 60,
):

    # keep the column names for later use
    cols = sdf.columns

    win = Window.partitionBy(identifier).orderBy(time)

    # add a column with the previous time stamp per identifier group
    sdf = sdf.withColumn("prev_timestamp", fn.lag(fn.col(time)).over(win))

    # add column with breaks (i.e, when the next flight ID should start)
    sdf = sdf.withColumn(
        "break",
        (
            fn.unix_timestamp(time, "yyyy-MM-dd HH:mm:ss.SSS")
            - fn.unix_timestamp("prev_timestamp", "yyyy-MM-dd HH:mm:ss.SSS")
            > threshold
        ).cast("integer"),
    ).na.fill({"break": 1})

    # assign a unique number to each row
    sdf = sdf.withColumn("idx", fn.monotonically_increasing_id())

    # create a column that is equal to 'idx' if 'break' == 1
    sdf = sdf.withColumn(
        "idx2",
        fn.when(fn.col("break") == 1, fn.col("idx")).otherwise(fn.lit(None)),
    )

    # as above, partition by identifier and order by time. Since there can be
    # multiple different IDs per partition, take the last non-NULL value.
    sdf = sdf.withColumn(
        "flight_id", fn.last(fn.col("idx2"), ignorenulls=True).over(win)
    )

    # drop temporary columns and return
    cols.extend(["flight_id"])
    return sdf.select(*cols)
