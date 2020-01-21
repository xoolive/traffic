from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxesSubplot

from cartotools.crs import *  # noqa: F401 F403
from cartotools.osm import location
from cartotools.osm.nominatim import Nominatim

from ..core.mixins import PointMixin, ShapelyMixin


def countries(**kwargs):
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


def rivers(**kwargs):
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


def lakes(**kwargs):
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


def ocean(**kwargs):
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


def _set_default_extent(self):
    """Helper for a default extent limited to the projection boundaries."""
    west, south, east, north = self.projection.boundary.bounds
    self.set_extent((west, east, south, north), crs=self.projection)


GeoAxesSubplot.set_default_extent = _set_default_extent


def _set_extent(self, shape, buffer: float = 0.01):
    if isinstance(shape, str):
        shape = location(shape)
    if isinstance(shape, ShapelyMixin):
        x1, x2, y1, y2 = shape.extent
        extent = (x1 - buffer, x2 + buffer, y1 - buffer, y2 + buffer)
        return self._set_extent(extent)
    if isinstance(shape, Nominatim):
        x1, x2, y1, y2 = shape.extent
        extent = (x1 - buffer, x2 + buffer, y1 - buffer, y2 + buffer)
        return self._set_extent(extent)
    self._set_extent(shape)


GeoAxesSubplot._set_extent = GeoAxesSubplot.set_extent
GeoAxesSubplot.set_extent = _set_extent


def _point(self):
    point = PointMixin()
    (point.longitude,), (point.latitude,) = self.shape.centroid.xy
    return point


setattr(Nominatim, "point", property(_point))
