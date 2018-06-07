try:
    from cartotools.crs import *  # noqa: F401 F403
    from cartotools.osm import location
except ImportError:
    # cartotools provides a few more basic projections
    from cartopy.crs import *  # noqa: F401 F403
    # Basic version of the complete cached requests included in cartotools
    from .location import location  # noqa: F401

from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.geoaxes import GeoAxesSubplot


def countries(**kwargs):
    params = {'category': 'cultural',
              'name': 'admin_0_countries',
              'scale': '10m',
              'edgecolor': '#524c50',
              'facecolor': 'none',
              'alpha': .5,
              **kwargs}
    return NaturalEarthFeature(**params)


def rivers(**kwargs):
    params = {'category': 'physical',
              'name': 'rivers_lake_centerlines',
              'scale': '10m',
              'edgecolor': '#226666',
              'facecolor': 'none',
              'alpha': .5,
              **kwargs}
    return NaturalEarthFeature(**params)


def lakes(**kwargs):
    params = {'category': 'physical',
              'name': 'lakes',
              'scale': '10m',
              'edgecolor': '#226666',
              'facecolor': '#226666',
              'alpha': .2,
              **kwargs}
    return NaturalEarthFeature(**params)


def ocean(**kwargs):
    params = {'category': 'physical',
              'name': 'ocean',
              'scale': '10m',
              'edgecolor': '#226666',
              'facecolor': '#226666',
              'alpha': .2,
              **kwargs}
    return NaturalEarthFeature(**params)


def _set_default_extent(self):
    """Helper for a default extent limited to the projection boundaries."""
    west, south, east, north = self.projection.boundary.bounds
    self.set_extent((west, east, south, north), crs=self.projection)


GeoAxesSubplot.set_default_extent = _set_default_extent
