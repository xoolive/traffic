from __future__ import annotations

from typing import Any, Iterable, Iterator, TypeVar

from typing_extensions import Annotated, Protocol

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


class ProgressbarType(Protocol):
    def __call__(
        self, iterable: Iterable[T], *args: Any, **kwargs: Any
    ) -> Iterator[T]: ...


class HasBounds(Protocol):
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...


## Types for physical units (impunity)

# sequence = Union[Sequence[float], npt.NDArray[np.float64]]
array = npt.NDArray[np.float64]

angle = Annotated[float, "degree"]
angle_array = Annotated[array, "degree"]

altitude = Annotated[float, "ft"]
altitude_array = Annotated[array, "ft"]

distance = Annotated[float, "nmi"]
distance_array = Annotated[array, "nmi"]

vertical_rate = Annotated[float, "ft/min"]
vertical_rate_array = Annotated[array, "ft/min"]

speed = Annotated[float, "kts"]
speed_array = Annotated[array, "kts"]

seconds = Annotated[float, "s"]
seconds_array = Annotated[array, "s"]
