from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class Implementation(Protocol):
    def __call__(self, df: "DataFrame", *args: Any, **kwds: Any) -> "DataFrame":
        ...


_implementations: dict[str, Implementation] = dict()


def spark_register(f: Implementation) -> Implementation:
    _implementations[f.__name__] = f
    return f


def get_alias(name: str) -> None | Implementation:
    return _implementations.get(name, None)


# TODO ensure strict typing
@spark_register
def query(df: "DataFrame", condition: str) -> "DataFrame":
    return df.filter(condition)
