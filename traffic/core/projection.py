import cartopy
import cartotools.crs
from cartotools.crs import *  # noqa: F401 F403

__all__ = [
    p
    for p in dir(cartotools.crs)
    if isinstance(getattr(cartotools.crs, p), type)
    and cartopy.crs.Projection in getattr(cartotools.crs, p).__mro__
]
