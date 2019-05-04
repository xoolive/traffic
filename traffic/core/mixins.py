import warnings
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, TypeVar, Union

import altair as alt
import pandas as pd
import pyproj
from cartopy import crs
from matplotlib.artist import Artist
from matplotlib.axes._subplots import Axes
from shapely.geometry import Point, base, mapping
from shapely.ops import transform

T = TypeVar("T", bound="DataFrameMixin")


class DataFrameMixin(object):

    """DataFrameMixin aggregates a pandas DataFrame and provides the same
    representation methods.

    """

    __slots__ = ()

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    @classmethod
    def from_file(
        cls: Type[T], filename: Union[Path, str], **kwargs
    ) -> Optional[T]:
        path = Path(filename)
        if path.suffixes in [[".pkl"], [".pkl", ".gz"]]:
            return cls(pd.read_pickle(path, **kwargs))
        if path.suffixes in [[".parquet"], [".parquet", ".gz"]]:
            return cls(pd.read_parquet(path, **kwargs))
        if path.suffixes in [[".json"], [".json", ".gz"]]:
            return cls(pd.read_json(path, **kwargs))
        if path.suffixes in [[".csv"], [".csv", ".gz"]]:
            return cls(pd.read_csv(path, **kwargs))
        if path.suffixes == [".h5"]:
            return cls(pd.read_hdf(path, **kwargs))
        return None

    # --- Special methods ---

    def _repr_html_(self):
        return self.data._repr_html_()

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self) -> int:
        return self.data.shape[0]

    # --- Redirected to pandas.DataFrame ---

    def to_pickle(self, filename: Union[str, Path], *args, **kwargs) -> None:
        self.data.to_pickle(filename, *args, **kwargs)

    def to_csv(self, filename: Union[str, Path], *args, **kwargs) -> None:
        self.data.to_csv(filename, *args, **kwargs)

    def to_hdf(self, filename: Union[str, Path], *args, **kwargs) -> None:
        self.data.to_hdf(filename, *args, **kwargs)

    def to_json(self, filename: Union[str, Path], *args, **kwargs) -> None:
        self.data.to_json(filename, *args, **kwargs)

    def to_parquet(self, filename: Union[str, Path], *args, **kwargs) -> None:
        self.data.to_parquet(filename, *args, **kwargs)

    def to_excel(self, filename: Union[str, Path], *args, **kwargs) -> None:
        self.data.to_excel(filename, *args, **kwargs)

    def sort_values(self: T, key: str) -> T:
        return self.__class__(self.data.sort_values(key))

    def query(self: T, query: str) -> T:
        return self.__class__(self.data.query(query))

    def drop(self: T, *args, **kwargs) -> T:
        return self.__class__(self.data.drop(*args, **kwargs))

    def rename(self: T, *args, **kwargs) -> T:
        return self.__class__(self.data.rename(*args, **kwargs))

    def pipe(self: T, func: Callable[..., T], *args, **kwargs) -> T:
        return func(self, *args, **kwargs)

    def fillna(self: T, *args, **kwargs) -> T:
        return self.__class__(self.data.fillna(*args, **kwargs))

    def groupby(self, *args, **kwargs):
        return self.data.groupby(*args, **kwargs)

    def assign(self: T, *args, **kwargs) -> T:
        return self.__class__(self.data.assign(*args, **kwargs))


class ShapelyMixin(object):

    """ShapelyMixin expects a shape attribute as a Geometry and provides methods
    consistent with GIS geometries.

    However no plot method is provided at this level because it depends on the
    nature of the shape.
    """

    __slots__ = ()

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

    def geojson(self):
        return mapping(self.shape)

    def geoencode(self) -> alt.vegalite.v2.api.Chart:
        return alt.Chart(alt.Data(values=self.geojson())).mark_geoshape(
            stroke="#aaaaaa", strokeWidth=1
        )

    @lru_cache()
    def project_shape(
        self, projection: Union[pyproj.Proj, crs.Projection, None] = None
    ) -> base.BaseGeometry:
        """Projection for a decent representation of the structure.

        By default, an equivalent projection is applied. Equivalent projections
        locally respect areas, which is convenient for the area attribute.
        """

        if self.shape is None:
            return None

        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

        if projection is None:
            bounds = self.bounds
            projection = pyproj.Proj(
                proj="aea",  # equivalent projection
                lat_1=bounds[1],
                lat_2=bounds[3],
                lat_0=(bounds[1] + bounds[3]) / 2,
                lon_0=(bounds[0] + bounds[2]) / 2,
            )
        projected_shape = transform(
            partial(
                pyproj.transform, pyproj.Proj(init="EPSG:4326"), projection
            ),
            self.shape,
        )

        if not projected_shape.is_valid:
            warnings.warn("The chosen projection is invalid for current shape")
        return projected_shape


class GeographyMixin(DataFrameMixin):
    """Adds Euclidean coordinates to a latitude/longitude DataFrame."""

    def compute_xy(
        self, projection: Union[pyproj.Proj, crs.Projection, None] = None
    ):
        """Computes x and y columns from latitudes and longitudes.

        The source projection is WGS84 (EPSG 4326).
        The default destination projection is a Lambert Conformal Conical
        projection centered on the data inside the dataframe.

        For consistency reasons with pandas DataFrame, a new Traffic structure
        is returned.

        """
        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

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

    def geoencode(self) -> alt.vegalite.v2.api.Chart:
        return (
            alt.Chart(
                self.data.query(
                    "latitude == latitude and longitude == longitude"
                )[["latitude", "longitude"]]
            )
            .encode(latitude="latitude", longitude="longitude")
            .mark_line()
        )


class PointMixin(object):

    latitude: float
    longitude: float
    altitude: float
    timestamp: datetime
    name: str

    @property
    def latlon(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    def plot(
        self, ax: Axes, text_kw=None, shift=dict(units="dots", x=15), **kwargs
    ) -> List[Artist]:

        if text_kw is None:
            text_kw = {}
        else:
            # since we may modify it, let's make a copy
            text_kw = {**text_kw}

        if "projection" in ax.__dict__ and "transform" not in kwargs:
            from cartopy.crs import PlateCarree
            from matplotlib.transforms import offset_copy

            kwargs["transform"] = PlateCarree()
            geodetic_transform = PlateCarree()._as_mpl_transform(ax)
            text_kw["transform"] = offset_copy(geodetic_transform, **shift)

        if "color" not in kwargs:
            kwargs["color"] = "black"

        if "s" not in text_kw:
            if hasattr(self, "callsign"):
                text_kw["s"] = getattr(self, "callsign")
            if hasattr(self, "name"):
                text_kw["s"] = getattr(self, "name")

        cumul: List[Artist] = []
        cumul.append(ax.scatter(self.longitude, self.latitude, **kwargs))
        cumul.append(ax.text(self.longitude, self.latitude, **text_kw))
        return cumul
