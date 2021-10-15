from typing import TYPE_CHECKING, Any, Tuple, Union

from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxesSubplot

from shapely.geometry.base import BaseGeometry

from ..core.mixins import PointMixin, ShapelyMixin

if TYPE_CHECKING:
    from cartes.osm import Nominatim


def countries(**kwargs: str) -> NaturalEarthFeature:

    params = {
        "category": "cultural",
        "name": "admin_0_countries",
        "scale": "10m",
        "edgecolor": "#524c50",
        "facecolor": "none",
        "alpha": 0.5,
        **kwargs,
    }
    return NaturalEarthFeature(**params)


def rivers(**kwargs: str) -> NaturalEarthFeature:

    params = {
        "category": "physical",
        "name": "rivers_lake_centerlines",
        "scale": "10m",
        "edgecolor": "#226666",
        "facecolor": "none",
        "alpha": 0.2,
        **kwargs,
    }
    return NaturalEarthFeature(**params)


def lakes(**kwargs: str) -> NaturalEarthFeature:

    params = {
        "category": "physical",
        "name": "lakes",
        "scale": "10m",
        "edgecolor": "#226666",
        "facecolor": "#226666",
        "alpha": 0.2,
        **kwargs,
    }
    return NaturalEarthFeature(**params)


def ocean(**kwargs: str) -> NaturalEarthFeature:

    params = {
        "category": "physical",
        "name": "ocean",
        "scale": "10m",
        "edgecolor": "#226666",
        "facecolor": "#226666",
        "alpha": 0.2,
        **kwargs,
    }
    return NaturalEarthFeature(**params)


def _set_default_extent(self: GeoAxesSubplot) -> None:
    """Helper for a default extent limited to the projection boundaries."""
    west, south, east, north = self.projection.boundary.bounds
    self.set_extent((west, east, south, north), crs=self.projection)


GeoAxesSubplot.set_default_extent = _set_default_extent


def _set_extent(
    self: GeoAxesSubplot,
    shape: Union[
        str, ShapelyMixin, "Nominatim", Tuple[float, float, float, float]
    ],
    buffer: float = 0.01,
) -> None:

    from cartes.osm import Nominatim

    if isinstance(shape, str):
        shape_ = Nominatim.search(shape)
        if shape_ is None:
            raise ValueError(f"'{shape}' not found on Nominatim")
        shape = shape_
    if isinstance(shape, ShapelyMixin):
        x1, x2, y1, y2 = shape.extent
        extent = (x1 - buffer, x2 + buffer, y1 - buffer, y2 + buffer)
        self._set_extent(extent)
        return
    if isinstance(shape, Nominatim):
        x1, x2, y1, y2 = shape.extent
        extent = (x1 - buffer, x2 + buffer, y1 - buffer, y2 + buffer)
        self._set_extent(extent)
        return
    self._set_extent(shape)
    return


# GeoAxesSubplot._set_extent = GeoAxesSubplot.set_extent
# GeoAxesSubplot.set_extent = _set_extent


def _point(self: BaseGeometry) -> PointMixin:
    point = PointMixin()
    (point.longitude,), (point.latitude,) = self.shape.centroid.xy
    return point


# TODO
# setattr(Nominatim, "point", property(_point))


def __getattr__(name: str) -> Any:

    import cartopy.crs

    if not name.startswith("_") and name in dir(cartopy.crs):
        return getattr(cartopy.crs, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")
