from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
ProgressbarType = Callable[[Iterable[T]], Iterable[T]]
