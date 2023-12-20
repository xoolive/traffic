from __future__ import annotations

from typing import Callable, Iterable, TypeVar

from typing_extensions import Annotated, Protocol

import numpy as np
import numpy.typing as npt

T = TypeVar("T")
ProgressbarType = Callable[[Iterable[T]], Iterable[T]]


class HasBounds(Protocol):
    bounds: tuple[float, float, float, float]


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
