# ruff: noqa: E501
from __future__ import annotations

import logging
import re
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Mapping,
    Sequence,
    TypedDict,
)

from py7zr import SevenZipFile
from rich.box import SIMPLE_HEAVY
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
from typing_extensions import Self

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, base, mapping
from shapely.ops import transform

from . import types as tt

if TYPE_CHECKING:
    import altair as alt
    import xarray
    from cartopy import crs
    from cartopy.mpl.geoaxes import GeoAxes
    from ipyleaflet import Marker as LeafletMarker
    from matplotlib.artist import Artist


# T = TypeVar("T", bound="DataFrameMixin")
# G = TypeVar("G", bound="GeoDBMixin")


_log = logging.getLogger(__name__)


class LatLonDict(TypedDict):
    lat: float
    lon: float


class DataFrameMixin(object):
    """DataFrameMixin aggregates a pandas DataFrame and provides the same
    representation methods.

    """

    __slots__ = ()

    table_options: ClassVar[dict[str, Any]] = dict(
        show_lines=False, box=SIMPLE_HEAVY
    )
    max_rows: int = 10
    columns_options: None | dict[str, dict[str, Any]] = None
    _obfuscate: None | list[str] = None

    def __init__(self, data: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
        self.data: pd.DataFrame = data  # type: ignore

    def __sizeof__(self) -> int:
        return int(self.data.memory_usage().sum())

    @classmethod
    def from_file(cls, filename: str | Path, **kwargs: Any) -> Self:
        """Read data from various formats.

        This class method dispatches the loading of data in various format to
        the proper ``pandas.read_*`` method based on the extension of the
        filename. Potential compression of the file is inferred by pandas itself
        based on the extension.

        - .pkl.* or .parquet.* dispatch to :func:`pandas.read_pickle`;
        - .parquet.* dispatch to :func:`pandas.read_parquet`;
        - .feather.* dispatch to :func:`pandas.read_feather`;
        - .json.* dispatch to :func:`pandas.read_json`;
        - .jsonl dispatch to a specific function (output of jet1090);
        - .csv.* dispatch to :func:`pandas.read_csv`;
        - .h5.* dispatch to :func:`pandas.read_hdf`.

        Other extensions return None.  Specific arguments may be passed to the
        underlying ``pandas.read_*`` method with the kwargs argument.

        Example usage:

        >>> from traffic.core import Traffic
        >>> t = Traffic.from_file(filename)
        """
        path = Path(filename)

        if path.suffix == (".7z"):
            with SevenZipFile(path) as archive:
                if (files := archive.readall()) is None:
                    raise FileNotFoundError(f"Empty archive {path}")
                for name, io in files.items():
                    if name.endswith(".jsonl"):
                        return cls(pd.read_json(io, lines=True, **kwargs))
                raise FileNotFoundError(f"Empty archive {path}")

        if ".pkl" in path.suffixes or ".pickle" in path.suffixes:
            return cls(pd.read_pickle(path, **kwargs))
        if ".parquet" in path.suffixes:
            return cls(pd.read_parquet(path, **kwargs))
        if ".feather" in path.suffixes:  # coverage: ignore
            return cls(pd.read_feather(path, **kwargs))
        if ".json" in path.suffixes:
            return cls(pd.read_json(path, **kwargs))
        if ".jsonl" in path.suffixes:
            return cls(pd.read_json(path, lines=True, **kwargs))
        if ".csv" in path.suffixes:
            return cls(pd.read_csv(path, **kwargs))
        if ".h5" == path.suffixes[-1]:  # coverage: ignore
            return cls(pd.read_hdf(path, **kwargs))

        raise FileNotFoundError(path)

    # --- Special methods ---

    def _repr_html_(self) -> str:
        return self.data._repr_html_()  # type: ignore

    # def __repr__(self) -> str:
    #    return repr(self.data)

    def __len__(self) -> int:
        return self.data.shape[0]  # type: ignore

    def __getitem__(self, index: Any) -> Any:
        return self.data.iloc[index]

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        my_table = Table(**self.table_options)

        if self.columns_options is None:
            self.columns_options = dict(  # type: ignore
                (column, dict()) for column in self.data.columns
            )

        for column, opts in self.columns_options.items():
            my_table.add_column(column, **opts)

        # This is only for documentation purposes, shouldn't be considered for
        # real-life code
        data = self.data[: self.max_rows]
        if self._obfuscate:
            for column in self._obfuscate:
                data = data.assign(**{column: "xxxxxx"})

        for _, elt in data.iterrows():
            my_table.add_row(
                *list(
                    format(
                        elt.get(column, ""),
                        ".4g"
                        if isinstance(elt.get(column, ""), Real)
                        and not isinstance(elt.get(column, ""), Integral)
                        else "",
                    )
                    for column in self.columns_options
                )
            )

        yield my_table

        delta = self.data.shape[0] - self.max_rows
        if delta > 0:
            yield f"... ({delta} more entries)"

    # --- Redirected to pandas.DataFrame ---

    def assign(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.assign` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.assign(*args, **kwargs))

    def convert_dtypes(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.convert_dtypes` method to
        the underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.convert_dtypes(*args, **kwargs))

    def drop(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.drop` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.drop(*args, **kwargs))

    def drop_duplicates(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.drop_duplicates` method to
        the underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.drop_duplicates(*args, **kwargs))

    def fillna(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.fillna` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.fillna(*args, **kwargs))

    def groupby(
        self, *args: Any, **kwargs: Any
    ) -> pd.core.groupby.generic.DataFrameGroupBy:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.groupby` method to the
        underlying pandas DataFrame.
        """
        return self.data.groupby(*args, **kwargs)

    def merge(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.merge` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.merge(*args, **kwargs))

    def query(self, query_str: str, *args: Any, **kwargs: Any) -> None | Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.query` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        df = self.data.query(query_str, *args, **kwargs)
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def rename(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.rename` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.rename(*args, **kwargs))

    def replace(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.replace` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.replace(*args, **kwargs))

    def reset_index(self, *args: Any, **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.reset_index` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.reset_index(*args, **kwargs))

    def sort_values(self, by: str | Sequence[str], **kwargs: Any) -> Self:
        """
        Applies the Pandas :meth:`~pandas.DataFrame.sort_values` method to the
        underlying pandas DataFrame and get the result back in the same
        structure.
        """
        return self.__class__(self.data.sort_values(by, **kwargs))

    def to_pickle(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to pickle format.

        Options can be passed to :meth:`pandas.DataFrame.to_pickle`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_pickle(filename, *args, **kwargs)

    def to_csv(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to CSV format.

        Options can be passed to :meth:`pandas.DataFrame.to_csv`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_csv(filename, *args, **kwargs)

    def to_hdf(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to HDF format.

        Options can be passed to :meth:`pandas.DataFrame.to_hdf`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_hdf(filename, *args, **kwargs)

    def to_json(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to JSON format.

        Options can be passed to :meth:`pandas.DataFrame.to_json`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_json(filename, *args, **kwargs)

    def to_parquet(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to parquet format.

        Options can be passed to :meth:`pandas.DataFrame.to_parquet`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_parquet(filename, *args, **kwargs)

    def to_feather(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to feather format.

        Options can be passed to :meth:`pandas.DataFrame.to_feather`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_feather(filename, *args, **kwargs)

    def to_excel(
        self, filename: str | Path, *args: Any, **kwargs: Any
    ) -> None:  # coverage: ignore
        """Exports to Excel format.

        Options can be passed to :meth:`pandas.DataFrame.to_excel`
        as args and kwargs arguments.

        Read more: :ref:`How to export and store trajectory and airspace data?`

        """
        self.data.to_excel(filename, *args, **kwargs)


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
    def bounds(self) -> tuple[float, float, float, float]:
        """Returns the bounds of the (bounding box of the) shape.
        Bounds are given in the following order in the origin crs:
        (west, south, east, north)
        """
        return self.shape.bounds  # type: ignore

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Returns the extent of the (bounding box of the) shape.

        .. note::
            When plotting with Matplotlib and Cartopy, the extent property is
            convenient in the following use case:

            >>> ax.set_extent(obj.extent)

        :return:
            Extent is given in the following order in the origin crs:
            (west, east, south, north)

        """
        west, south, east, north = self.bounds
        return west, east, south, north

    @property
    def centroid(self) -> Point:
        """Returns the centroid of the shape as a shapely Point."""
        return self.shape.centroid

    @property
    def area(self) -> float:
        """Returns the area of the shape, in square meters.
        The shape is projected to an equivalent local projection before
        computing a value.
        """
        return self.project_shape().area  # type: ignore

    # --- Representations ---

    @lru_cache()
    def _repr_svg_(self) -> None | str:
        if self.shape.is_empty:
            return None
        project = self.project_shape()
        if project is not None:
            return project._repr_svg_()  # type: ignore
        return None

    def _repr_html_(self) -> str:
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return no_wrap_div.format(self._repr_svg_())

    def geojson(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Returns the GeoJSON representation of the shape as a Dict.
        The transformation is delegated to shapely ``mapping`` method.
        """
        return mapping(self.shape)  # type: ignore

    def geoencode(
        self, **kwargs: Any
    ) -> "alt.LayerChart | alt.Chart":  # coverage: ignore
        """Returns an `altair <http://altair-viz.github.io/>`_ encoding of the
        shape to be composed in an interactive visualization. Specific plot
        features, such as line widths, can be passed with the kwargs argument.
        See `documentation
        <https://altair-viz.github.io/user_guide/marks.html>`_.
        """
        import altair as alt

        data = alt.Data(values=self.geojson())  # type: ignore
        return alt.Chart(data).mark_geoshape(stroke="#aaaaaa", **kwargs)

    def project_shape(
        self, projection: None | pyproj.Proj | "crs.Projection" = None
    ) -> base.BaseGeometry:
        """Returns a projected representation of the shape.

        :param projection: By default (None), an equivalent projection is
            applied. Equivalent projections locally respect areas, which is
            convenient for the area attribute.

        """

        from cartopy import crs

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

        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), projection, always_xy=True
        )
        projected_shape = transform(
            transformer.transform,
            self.shape,
        )

        if not projected_shape.is_valid:
            _log.warning("The chosen projection is invalid for current shape")
        return projected_shape


class GeographyMixin(DataFrameMixin):
    """Adds Euclidean coordinates to a latitude/longitude DataFrame."""

    __slots__ = ()

    def projection(self, proj: str = "lcc") -> pyproj.Proj:
        return pyproj.Proj(
            proj=proj,
            ellps="WGS84",
            lat_1=self.data.latitude.min(),
            lat_2=self.data.latitude.max(),
            lat_0=self.data.latitude.mean(),
            lon_0=self.data.longitude.mean(),
        )

    def compute_xy(
        self, projection: None | pyproj.Proj | "crs.Projection" = None
    ) -> Self:
        """Enrich the structure with new x and y columns computed through a
        projection of the latitude and longitude columns.

        The source projection is WGS84 (EPSG 4326).
        The default destination projection is a Lambert Conformal Conical
        projection centred on the data inside the dataframe.

        Other valid projections are available:

        - as ``pyproj.Proj`` objects;
        - as ``cartopy.crs.Projection`` objects.
        """
        from cartopy import crs

        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

        if projection is None:
            projection = self.projection(proj="lcc")

        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), projection, always_xy=True
        )
        x, y = transformer.transform(
            self.data.longitude.to_numpy(),
            self.data.latitude.to_numpy(),
        )

        return self.__class__(self.data.assign(x=x, y=y))

    def compute_latlon_from_xy(
        self, projection: pyproj.Proj | crs.Projection
    ) -> Self:
        """Enrich a DataFrame with new longitude and latitude columns computed
        from x and y columns.

        .. warning::

            Make sure to use as source projection the one used to compute
            ``'x'`` and ``'y'`` columns in the first place.
        """

        from cartopy import crs

        if not set(["x", "y"]).issubset(set(self.data.columns)):
            raise ValueError("DataFrame should contains 'x' and 'y' columns.")

        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

        transformer = pyproj.Transformer.from_proj(
            projection, pyproj.Proj("epsg:4326"), always_xy=True
        )
        lon, lat = transformer.transform(
            self.data.x.to_numpy(),
            self.data.y.to_numpy(),
        )

        return self.assign(latitude=lat, longitude=lon)

    def agg_xy(
        self,
        resolution: None | dict[str, float],
        projection: None | pyproj.Proj | "crs.Projection" = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Aggregates values of a traffic over a grid of x/y, with x and y
        computed by :meth:`~traffic.core.GeographyMixin.compute_xy()`.

        :param resolution: The resolution of the grid is passed as a dictionary
             parameter. By default, the grid is made by rounding x and y to the
             lower ten kilometer values.
             ``dict(x=5000, y=3000)`` will take 1 value per 5000 meters for x
             (10000, 15000, 20000, ...) and 1 value per 3000 meters for y (9000,
             12000, 15000, 18000, 20000, ...).

        :param projection: is used to compute the x and y values.

        The kwargs specifies how to aggregate values:

        - ``altitude="mean"`` would average all values in the given cell;
        - ``timestamp="count"`` would return the number of samples per cell;
        - ``icao24="nunique"`` would return the number of different aircraft
          int the given cell.

        :return: a :class:`~pandas.DataFrame` indexed over x and y values. It is
            conveniently chainable with the ``.to_xarray()`` method in order to
            plot density heatmaps.

        Example usage:

        .. code:: python

            belevingsvlucht.agg_xy(
                resolution=dict(x=3e3, y=3e3),
                vertical_rate="mean",
                timestamp="count"
            )
        """
        if resolution is None:
            resolution = dict(x=1e4, y=1e4)

        if len(kwargs) is None:
            raise ValueError(
                "Specify parameters to aggregate, "
                "e.g. altitude='mean' or icao24='nunique'"
            )

        r_x = resolution.get("x", None)
        r_y = resolution.get("y", None)

        if r_x is None or r_y is None:
            raise ValueError("Specify a resolution for x and y")

        data = (
            self.compute_xy(projection)
            .assign(
                x=lambda elt: (elt.x // r_x) * r_x,
                y=lambda elt: (elt.y // r_y) * r_y,
            )
            .groupby(["x", "y"])
            .agg(kwargs)
        )

        return data

    def geoencode(self, **kwargs: Any) -> "alt.Chart":  # coverage: ignore
        """Returns an `altair <http://altair-viz.github.io/>`_ encoding of the
        shape to be composed in an interactive visualization. Specific plot
        features, such as line widths, can be passed with the kwargs argument.
        See `documentation
        <https://altair-viz.github.io/user_guide/marks.html>`_.
        """
        import altair as alt

        return (
            alt.Chart(
                self.data.query("latitude.notnull() and longitude.notnull()")
            )
            .encode(latitude="latitude", longitude="longitude")
            .mark_line(**kwargs)
        )

    def interpolate_grib(
        self, wind: "xarray.Dataset", features: list[str] = ["u", "v"]
    ) -> Self:
        from openap import aero
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        projection: pyproj.Proj = self.projection("lcc")
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), projection, always_xy=True
        )

        west, east = self.data.longitude.min(), self.data.longitude.max()
        longitude_index = wind.longitude.to_numpy()
        margin = np.diff(longitude_index).max()
        longitude_index = longitude_index[
            np.where(
                (longitude_index >= west - margin)
                & (longitude_index <= east + margin)
            )
        ]

        south, north = self.data.latitude.min(), self.data.latitude.max()
        latitude_index = wind.latitude.to_numpy()
        margin = np.diff(latitude_index).max()
        latitude_index = latitude_index[
            np.where(
                (latitude_index >= south - margin)
                & (latitude_index <= north + margin)
            )
        ]

        timestamp = self.data.timestamp.dt.tz_convert("utc")
        start, stop = timestamp.min(), timestamp.max()
        time_index = wind.time.to_numpy()
        margin = np.diff(time_index).max()
        time_index = time_index[
            np.where(
                (time_index >= start.tz_localize(None) - margin)
                & (time_index <= stop.tz_localize(None) + margin)
            )
        ]

        idx_max = 1 + np.sum(
            aero.h_isa(wind.isobaricInhPa.to_numpy() * 100)
            < self.data.altitude.max() * aero.ft
        )
        isobaric_index = wind.isobaricInhPa.to_numpy()[:idx_max]

        wind_df = (
            wind.sel(
                longitude=longitude_index,
                latitude=latitude_index,
                time=time_index,
                isobaricInhPa=isobaric_index,
            )
            .to_dataframe()
            .reset_index()
            .assign(h=lambda df: aero.h_isa(df.isobaricInhPa * 100))
        )

        wind_x, wind_y = transformer.transform(
            wind_df.longitude.to_numpy(),
            wind_df.latitude.to_numpy(),
        )
        wind_xy = wind_df.assign(x=wind_x, y=wind_y)

        model = make_pipeline(PolynomialFeatures(2), Ridge())
        model.fit(wind_xy[["x", "y", "h"]], wind_xy[list(features)])

        poly_features = [
            s.replace("^", "**").replace(" ", "*")
            for s in model["polynomialfeatures"].get_feature_names()
        ]
        ridges = model["ridge"].coef_

        x, y = transformer.transform(
            self.data.longitude.to_numpy(),
            self.data.latitude.to_numpy(),
        )
        h = self.data.altitude.to_numpy() * aero.ft

        return self.assign(
            **dict(
                (
                    name,
                    sum(
                        [
                            eval(f, {}, {"x0": x, "x1": y, "x2": h}) * c
                            for (f, c) in zip(poly_features, ridge_coefficients)
                        ]
                    ),
                )
                for name, ridge_coefficients in zip(features, ridges)
            )
        )


class GeoDBMixin(DataFrameMixin):
    _extent: None | tuple[float, float, float, float] = None
    __slots__ = ()

    def extent(
        self,
        extent: str | ShapelyMixin | tuple[float, float, float, float],
        buffer: float = 0.5,
    ) -> None | Self:
        """
        Selects the subset of data inside the given extent.

        :param extent:
            The parameter extent may be passed as:

            - a string to query OSM Nominatim service;
            - the result of an OSM Nominatim query
              (:class:`~traffic.core.mixins.ShapelyMixin`);
            - any kind of shape (:class:`~traffic.core.mixins.ShapelyMixin`,
              including :class:`~traffic.core.Airspace`);
            - extent in the order:  (west, east, south, north)

        :param buffer:
            As the extent of a given shape may be a little too strict to catch
            elements we may expect when we look into an area, the buffer
            parameter (by default, 0.5 degree) helps enlarging the area of
            interest.

        This works with databases like
        :class:`~traffic.data.basic.airways.Airways`,
        :class:`~traffic.data.basic.airports.Airports` or
        :class:`~traffic.data.basic.navaid.Navaids`.

        >>> from traffic.data import airways
        >>> airways.extent(eurofirs['LFBB'])  # doctest: +SKIP
          route    id   navaid   latitude    longitude
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          A25      6    GODAN    47.64       -1.96
          A25      7    TMA28    47.61       -1.935
          A25      8    NTS      47.16       -1.613
          A25      9    TIRAV    46.6        -1.391
          A25      10   LUSON    46.5        -1.351
          A25      11   OLERO    45.97       -1.15
          A25      12   MAREN    45.73       -1.062
          A25      13   ROYAN    45.67       -1.037
          A25      14   BMC      44.83       -0.7211
          A25      15   SAU      44.68       -0.1529
         ... (703 more lines)

        >>> from traffic.data import airports
        >>> airports.extent("Bornholm")  # doctest: +SKIP
         name                 country   icao      iata   latitude   longitude
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         Bodilsker Airstrip   Denmark   DK-0027   nan    55.06      15.05
         Bornholm Airport     Denmark   EKRN      RNN    55.06      14.76
         Ro Airport           Denmark   EKRR      nan    55.21      14.88

        >>> from traffic.data import navaids
        >>> navaids['ZUE']
        Navaid('ZUE', type='NDB', latitude=30.9, longitude=20.06833333, altitude=0.0, description='ZUEITINA NDB', frequency='369.0kHz')
        >>> navaids.extent('Switzerland')['ZUE']
        Navaid('ZUE', type='VOR', latitude=47.59216667, longitude=8.81766667, altitude=1730.0, description='ZURICH EAST VOR-DME', frequency='110.05MHz')

        """
        from cartes.osm import Nominatim

        _extent = (0.0, 0.0, 0.0, 0.0)

        if isinstance(extent, str):
            loc = Nominatim.search(extent)
            if loc is not None:
                _extent = loc.extent
        if isinstance(extent, ShapelyMixin):
            _extent = extent.extent
        if isinstance(extent, Nominatim):
            _extent = extent.extent
        if isinstance(extent, (tuple, list)):
            _extent = extent

        west, east, south, north = _extent

        output = self.query(
            f"{south - buffer} <= latitude <= {north + buffer} and "
            f"{west - buffer} <= longitude <= {east + buffer}"
        )

        if output is not None:
            output._extent = _extent

        return output

    def geoencode(self, **kwargs: Any) -> "alt.Chart":  # coverage: ignore
        """Returns an `altair <http://altair-viz.github.io/>`_ encoding of the
        shape to be composed in an interactive visualization.
        """
        import altair as alt

        return (
            alt.Chart(self.data)
            .mark_circle(**kwargs)
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                size=alt.value(3),
                color=alt.value("steelblue"),
            )
        )


class PointMixin:
    latitude: tt.angle
    longitude: tt.angle
    altitude: tt.altitude
    track: tt.angle
    timestamp: pd.Timestamp
    name: str

    @property
    def latlon(self) -> tuple[float, float]:
        """A tuple for latitude and longitude, in degrees, in this order."""
        return (self.latitude, self.longitude)

    @property
    def latlon_dict(self) -> LatLonDict:
        """A tuple for latitude and longitude, in degrees, in this order."""
        return dict(lat=self.latitude, lon=self.longitude)

    def leaflet(self, **kwargs: Any) -> "LeafletMarker":
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

    def plot(
        self,
        ax: "GeoAxes",
        text_kw: None | Mapping[str, Any] = None,
        shift: None | Mapping[str, Any] = None,
        **kwargs: Any,
    ) -> list["Artist"]:  # coverage: ignore
        if shift is None:
            shift = dict(units="dots", x=15)

        if text_kw is None:
            text_kw = dict()
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

        cumul: list["Artist"] = []
        cumul.append(ax.scatter(self.longitude, self.latitude, **kwargs))

        west, east, south, north = ax.get_extent(PlateCarree())
        if west <= self.longitude <= east and south <= self.latitude <= north:
            cumul.append(ax.text(self.longitude, self.latitude, **text_kw))

        return cumul


class FormatMixin(object):  # coverage: ignore
    def __format__(self, pattern: str) -> str:
        if pattern == "":
            return repr(self)
        for match in re.finditer(r"%(\w+)", pattern):
            pattern = pattern.replace(
                match.group(0), getattr(self, match.group(1), "")
            )
        return pattern


class _HBox(object):  # coverage: ignore
    def __init__(self, *args: Any) -> None:
        self.elts = args

    def _repr_html_(self) -> str:
        return "".join(
            f"""
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        vertical-align: top;'>
        {elt._repr_html_()}
    </div>
    """
            for elt in self.elts
        )

    def __or__(self, other: Any) -> "_HBox":
        if isinstance(other, _HBox):
            return _HBox(*self.elts, *other.elts)
        else:
            return _HBox(*self.elts, other)

    def __ror__(self, other: Any) -> "_HBox":
        if isinstance(other, _HBox):
            return _HBox(*other.elts, *self.elts)
        else:
            return _HBox(other, *self.elts)


class HBoxMixin(object):  # coverage: ignore
    """Enables a | operator for placing representations next to each other."""

    def __or__(self, other: Any) -> _HBox:
        if isinstance(other, _HBox):
            return _HBox(self, *other.elts)
        else:
            return _HBox(self, other)


def patch_leaflet() -> None:
    from ..visualize.leaflet import point_leaflet

    PointMixin.leaflet = point_leaflet  # type: ignore


try:
    patch_leaflet()
except Exception:
    pass
