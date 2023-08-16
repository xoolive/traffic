from __future__ import annotations

from typing import Callable, Iterable, TypeVar

from typing_extensions import Protocol

T = TypeVar("T")
ProgressbarType = Callable[[Iterable[T]], Iterable[T]]


class HasBounds(Protocol):
    bounds: tuple[float, float, float, float]
