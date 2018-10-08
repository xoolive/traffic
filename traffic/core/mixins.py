from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Tuple, Union

from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes

import pandas as pd
import pyproj
from shapely.geometry import Point, base
from shapely.ops import transform


class DataFrameMixin(object):

    """DataFrameMixin aggregates a pandas DataFrame and provides the same
    representation methods.

    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    @classmethod
    def from_file(cls, filename: Union[Path, str]):
        path = Path(filename)
        if path.suffixes in [[".pkl"], [".pkl", ".gz"]]:
            return cls(pd.read_pickle(path))
        if path.suffixes == [".csv"]:
            return cls(pd.read_csv(path))
        if path.suffixes == [".h5"]:
            return cls(pd.read_hdf(path))
        return None

    # --- Special methods ---

    def _repr_html_(self):
        return self.data._repr_html_()

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self) -> int:
        return self.data.shape[0]

    # --- Redirected to pandas.DataFrame ---

    def to_pickle(self, filename: Union[str, Path]) -> None:
        self.data.to_pickle(filename)

    def to_csv(self, filename: Union[str, Path]) -> None:
        self.data.to_csv(filename)

    def to_hdf(self, filename: Union[str, Path]) -> None:
        self.data.to_hdf(filename)

    def to_json(self, filename: Union[str, Path]) -> None:
        self.data.to_json(filename)

    def to_excel(self, filename: Union[str, Path]) -> None:
        self.data.to_excel(filename)

    def sort_values(self, key: str):
        return self.__class__(self.data.sort_values(key))

    def query(self, query: str):
        return self.__class__(self.data.query(query))

    def groupby(self, *args, **kwargs):
        return self.data.groupby(*args, **kwargs)

    def assign(self, *args, **kwargs):
        return self.__class__(self.data.assign(*args, **kwargs))


class ShapelyMixin(object):

    """ShapelyMixin expects a shape attribute as a Geometry and provides methods
    consistent with GIS geometries.

    However no plot method is provided at this level because it depends on the
    nature of the shape.
    """

    shape: base.BaseGeometry

    # --- Properties ---

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns the bounds of the shape.
        Bounds are given in the following order in the origin crs:
        west, south, east, north
        """
        return self.shape.bounds

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Returns the extent of the shape.
        Extent is given in the following order in the origin crs:
        west, east, south, north

        This method is convenient for the ax.set_extent method
        """
        west, south, east, north = self.bounds
        return west, east, south, north

    @property
    def centroid(self) -> Point:
        """Returns the centroid of the shape."""
        return self.shape.centroid

    @property
    def area(self) -> float:
        """Returns the area of the shape, in square meters."""
        return self.project_shape().area

    # --- Representations ---

    def _repr_svg_(self):
        project = self.project_shape()
        if project is not None:
            return project._repr_svg_()

    def _repr_html_(self) -> str:
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return no_wrap_div.format(self._repr_svg_())

    @lru_cache()
    def project_shape(
        self, projection: pyproj.Proj = None
    ) -> base.BaseGeometry:
        """Projection for a decent representation of the structure.

        By default, an equivalent projection is applied. Equivalent projections
        locally respect areas, which is convenient for the area attribute.
        """

        if self.shape is None:
            return None

        if projection is None:
            bounds = self.bounds
            projection = pyproj.Proj(
                proj="aea",  # equivalent projection
                lat1=bounds[1],
                lat2=bounds[3],
                lon1=bounds[0],
                lon2=bounds[2],
            )
        return transform(
            partial(
                pyproj.transform, pyproj.Proj(init="EPSG:4326"), projection
            ),
            self.shape,
        )


class GeographyMixin(object):
    """Adds Euclidean coordinates to a latitude/longitude DataFrame."""

    data: pd.DataFrame

    def compute_xy(self, projection: pyproj.Proj = None):
        """Computes x and y columns from latitudes and longitudes.

        The source projection is WGS84 (EPSG 4326).
        The default destination projection is a Lambert Conformal Conical
        projection centered on the data inside the dataframe.

        For consistency reasons with pandas DataFrame, a new Traffic structure
        is returned.

        """
        if projection is None:
            projection = pyproj.Proj(
                proj="lcc",
                lat_1=self.data.latitude.min(),
                lat_2=self.data.latitude.max(),
                lat_0=self.data.latitude.mean(),
                lon_0=self.data.longitude.mean(),
            )

        x, y = pyproj.transform(
            pyproj.Proj(init="EPSG:4326"),
            projection,
            self.data.longitude.values,
            self.data.latitude.values,
        )

        return self.__class__(self.data.assign(x=x, y=y))


class PointMixin(object):

    latitude: float
    longitude: float
    altitude: float
    timestamp: datetime

    def plot(
        self, ax: Axes, text_kw=None, shift=dict(units="dots", x=15), **kwargs
    ):

        if text_kw is None:
            text_kw = {}

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            from cartopy.crs import PlateCarree
            from matplotlib.transforms import offset_copy

            kwargs["transform"] = PlateCarree()
            geodetic_transform = PlateCarree()._as_mpl_transform(ax)
            text_kw["transform"] = offset_copy(geodetic_transform, **shift)

        if "color" not in kwargs:
            kwargs["color"] = "black"

        if 's' not in text_kw:
            text_kw['s'] = getattr(self, 'callsign', '')

        cumul: List[Artist] = []
        cumul.append(ax.scatter(self.longitude, self.latitude, **kwargs))
        cumul.append(ax.text(self.longitude, self.latitude, **text_kw))
        return cumul
