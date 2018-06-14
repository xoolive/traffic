import logging
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import pyproj
from shapely.ops import transform


class DataFrameMixin(object):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    @classmethod
    def from_file(
        cls, filename: Union[Path, str]
    ) -> Optional["DataFrameMixin"]:
        path = Path(filename)
        if path.suffixes == [".pkl"]:
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

    def query(self, query: str) -> "DataFrameMixin":
        return self.__class__(self.data.query(query))

    def groupby(self, *args, **kwargs):
        return self.data.groupby(*args, **kwargs)


class ShapelyMixin(object):

    # --- Properties ---

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        # just natural!
        return self.shape.bounds  # type: ignore

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        # convenient for ax.set_extent
        west, south, east, north = self.bounds
        return west, east, south, north

    @property
    def centroid(self):
        return self.shape.centroid

    @property
    def area(self) -> float:
        return self.project_shape().area

    # --- Representations ---

    def _repr_svg_(self):
        project = self.project_shape()
        if project is not None:
            return project._repr_svg_()

    def _repr_html_(self):
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return no_wrap_div.format(self._repr_svg_())

    @lru_cache()
    def project_shape(self, projection=None):
        """Projection for a decent representation of the structure.

        By default, an equivalent projection is applied. Equivalent projections
        locally respect areas, which is convenient for the area item.
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

    # Nothing special made on plots because it really depends on the nature of
    # the geometry of the shape held in the structure.


class GeographyMixin(object):
    def compute_xy(self, projection=None):
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
