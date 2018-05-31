import logging
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pyproj
from shapely.ops import transform


class DataFrameMixin(object):

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    def _repr_html_(self):
        return self.data._repr_html_()

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self) -> int:
        return self.data.shape[0]

    def to_pickle(self, filename: str) -> None:
        self.data.to_pickle(filename)

    def to_csv(self, filename: str) -> None:
        self.data.to_csv(filename)

    def to_hdf(self, filename: str) -> None:
        self.data.to_hdf(filename)

    def to_json(self, filename: str) -> None:
        self.data.to_json(filename)

    def to_excel(self, filename: str) -> None:
        self.data.to_excel(filename)

    @classmethod
    def from_file(cls, filename: str) -> Optional['DataFrameMixin']:
        path = Path(filename)
        if path.suffixes == ['.pkl']:
            return cls(pd.read_pickle(path))
        logging.warn(f"Suffix {''.join(path.suffixes)} not supported")
        return None

    def query(self, query: str) -> 'DataFrameMixin':
        return self.__class__(self.data.query(query))


class ShapelyMixin(object):

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

    def _repr_svg_(self):
        return self.project_shape()._repr_svg_()

    def _repr_html_(self):
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return no_wrap_div.format(self._repr_svg_())

    @property
    def projected_shape(self):
        return self.project_shape()

    @property
    def area(self) -> float:
        return self.projected_shape.area

    @lru_cache()
    def project_shape(self, projection=None):
        if projection is None:
            bounds = self.bounds
            projection = pyproj.Proj(proj='aea',  # equivalent projection
                                     lat1=bounds[1], lat2=bounds[3],
                                     lon1=bounds[0], lon2=bounds[2])
        return transform(partial(pyproj.transform,
                                 pyproj.Proj(init='EPSG:4326'),
                                 projection), self.shape)

