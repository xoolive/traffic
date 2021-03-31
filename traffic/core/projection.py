import cartes.crs
import cartopy
from cartes.crs import *  # noqa: F401 F403

__all__ = [
    p
    for p in dir(cartes.crs)
    if isinstance(getattr(cartes.crs, p), type)
    and cartopy.crs.Projection in getattr(cartes.crs, p).__mro__
]
