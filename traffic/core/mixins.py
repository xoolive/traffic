import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

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

    def __init__(self, data: pd.DataFrame, *args, **kwargs) -> None:
        self.data: pd.DataFrame = data

    def __sizeof__(self) -> int:
        return int(self.data.memory_usage().sum())

    @classmethod
    def from_file(
        cls: Type[T], filename: Union[Path, str], **kwargs
    ) -> Optional[T]:
        """Read data from various formats.

        This class method dispatches the loading of data in various format to
        the proper ``pandas.read_*`` method based on the extension of the
        filename.

        - .pkl and .pkl.gz dispatch to ``pandas.read_pickle``;
        - .parquet and .parquet.gz dispatch to ``pandas.read_parquet``;
        - .json and .json.gz dispatch to ``pandas.read_json``;
        - .csv and .csv.gz dispatch to ``pandas.read_csv``;
        - .h5 dispatch to ``pandas.read_hdf``.

        Other extensions return ``None``.
        Specific arguments may be passed to the underlying ``pandas.read_*``
        method with the kwargs argument.

        Example usage:

        >>> t = Traffic.from_file("data/sample_opensky.pkl")
        """
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

    def to_pickle(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> None:  # coverage: ignore
        """Exports to pickle format.

        Options can be passed to ``pandas.to_pickle``
        as args and kwargs arguments.

        Read more about export formats in the `Exporting and Storing data
        <./export.html>`_ section
        """
        self.data.to_pickle(filename, *args, **kwargs)

    def to_csv(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> None:  # coverage: ignore
        """Exports to CSV format.

        Options can be passed to ``pandas.to_csv``
        as args and kwargs arguments.

        Read more about export formats in the `Exporting and Storing data
        <./export.html>`_ section
        """
        self.data.to_csv(filename, *args, **kwargs)

    def to_hdf(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> None:  # coverage: ignore
        """Exports to HDF format.

        Options can be passed to ``pandas.to_hdf``
        as args and kwargs arguments.

        Read more about export formats in the `Exporting and Storing data
        <./export.html>`_ section
        """
        self.data.to_hdf(filename, *args, **kwargs)

    def to_json(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> None:  # coverage: ignore
        """Exports to JSON format.

        Options can be passed to ``pandas.to_json``
        as args and kwargs arguments.

        Read more about export formats in the `Exporting and Storing data
        <./export.html>`_ section
        """
        self.data.to_json(filename, *args, **kwargs)

    def to_parquet(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> None:  # coverage: ignore
        """Exports to parquet format.

        Options can be passed to ``pandas.to_parquet``
        as args and kwargs arguments.

        Read more about export formats in the `Exporting and Storing data
        <./export.html>`_ section
        """
        self.data.to_parquet(filename, *args, **kwargs)

    def to_excel(
        self, filename: Union[str, Path], *args, **kwargs
    ) -> None:  # coverage: ignore
        """Exports to Excel format.

        Options can be passed to ``pandas.to_excel``
        as args and kwargs arguments.

        Read more about export formats in the `Exporting and Storing data
        <./export.html>`_ section
        """
        self.data.to_excel(filename, *args, **kwargs)

    def sort_values(self: T, by: str, **kwargs) -> T:
        """
        Applies the Pandas ``sort_values()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.sort_values(by, **kwargs))

    def query(self: T, query_str: str, *args, **kwargs) -> Optional[T]:
        """
        Applies the Pandas ``query()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        df = self.data.query(query_str, *args, **kwargs)
        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def drop(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``drop()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.drop(*args, **kwargs))

    def rename(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``rename()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.rename(*args, **kwargs))

    def pipe(self: T, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Applies the Pandas ``pipe()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return func(self, *args, **kwargs)

    def fillna(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``fillna()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.fillna(*args, **kwargs))

    def groupby(self, *args, **kwargs):
        """
        Applies the Pandas ``groupby()`` method to the underlying pandas
        DataFrame.
        """
        return self.data.groupby(*args, **kwargs)

    def assign(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``assign()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.assign(*args, **kwargs))

    def drop_duplicates(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``drop_duplicates()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.drop_duplicates(*args, **kwargs))

    def merge(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``merge()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.merge(*args, **kwargs))

    def reset_index(self: T, *args, **kwargs) -> T:
        """
        Applies the Pandas ``reset_index()`` method to the underlying pandas
        DataFrame and get the result back in the same structure.
        """
        return self.__class__(self.data.reset_index(*args, **kwargs))


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
        """Returns the bounds of the (bounding box of the) shape.
        Bounds are given in the following order in the origin crs:
        (west, south, east, north)
        """
        return self.shape.bounds

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Returns the extent of the (bounding box of the) shape.
        Extent is given in the following order in the origin crs:
        (west, east, south, north)

        .. note::
            When plotting with Matplotlib and Cartopy, the extent property is
            convenient in the following use case:

            >>> ax.set_extent(obj.extent)
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
        return self.project_shape().area

    # --- Representations ---

    @lru_cache()
    def _repr_svg_(self):
        if self.shape.is_empty:
            return None
        project = self.project_shape()
        if project is not None:
            return project._repr_svg_()

    def _repr_html_(self):
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return no_wrap_div.format(self._repr_svg_())

    def geojson(self):
        """Returns the GeoJSON representation of the shape as a Dict.
        The transformation is delegated to shapely ``mapping`` method.
        """
        return mapping(self.shape)

    def geoencode(self, **kwargs) -> alt.Chart:  # coverage: ignore
        """Returns an `altair <http://altair-viz.github.io/>`_ encoding of the
        shape to be composed in an interactive visualization.
        Specific plot features, such as line widths, can be passed via **kwargs.
        See `documentation
        <https://altair-viz.github.io/user_guide/marks.html>`_.
        """
        return alt.Chart(alt.Data(values=self.geojson())).mark_geoshape(
            stroke="#aaaaaa", **kwargs
        )

    def project_shape(
        self, projection: Union[pyproj.Proj, crs.Projection, None] = None
    ) -> base.BaseGeometry:
        """Returns a projected representation of the shape.

        By default, an equivalent projection is applied. Equivalent projections
        locally respect areas, which is convenient for the area attribute.

        Other valid projections are available:

        - as ``pyproj.Proj`` objects;
        - as ``cartopy.crs.Projection`` objects.

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

        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), projection, always_xy=True
        )
        projected_shape = transform(
            transformer.transform,
            self.shape,
        )

        if not projected_shape.is_valid:
            warnings.warn("The chosen projection is invalid for current shape")
        return projected_shape


class GeographyMixin(DataFrameMixin):
    """Adds Euclidean coordinates to a latitude/longitude DataFrame."""

    def compute_xy(
        self: T, projection: Union[pyproj.Proj, crs.Projection, None] = None
    ) -> T:
        """Enrich the structure with new x and y columns computed through a
        projection of the latitude and longitude columns.

        The source projection is WGS84 (EPSG 4326).
        The default destination projection is a Lambert Conformal Conical
        projection centered on the data inside the dataframe.

        Other valid projections are available:

        - as ``pyproj.Proj`` objects;
        - as ``cartopy.crs.Projection`` objects.
        """

        if isinstance(projection, crs.Projection):
            projection = pyproj.Proj(projection.proj4_init)

        if projection is None:
            projection = pyproj.Proj(
                proj="lcc",
                ellps="WGS84",
                lat_1=self.data.latitude.min(),
                lat_2=self.data.latitude.max(),
                lat_0=self.data.latitude.mean(),
                lon_0=self.data.longitude.mean(),
            )

        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), projection, always_xy=True
        )
        x, y = transformer.transform(
            self.data.longitude.values,
            self.data.latitude.values,
        )

        return self.__class__(self.data.assign(x=x, y=y))

    def agg_xy(
        self,
        resolution: Union[Dict[str, float], None],
        projection: Union[pyproj.Proj, crs.Projection, None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Aggregates values of a traffic over a grid of x/y, with x and y
        computed by `traffic.core.GeographyMixin.compute_xy()`.

        The resolution of the grid is passed as a dictionary parameter.
        By default, the grid is made by rounding x and y to the lower ten
        kilometer values. ``dict(x=5000, y=3000)`` will take 1 value per 5000
        meters for x (10000, 15000, 20000, ...) and 1 value per 3000 meters for
        y (9000, 12000, 15000, 18000, 20000, ...).

        The kwargs specifies how to aggregate values:

        - ``altitude="mean"`` would average all values in the given cell;
        - ``timestamp="count"`` would return the number of samples per cell;
        - ``icao24="nunique"`` would return the number of different aircraft
          int the given cell.

        The returned pandas DataFrame is indexed over x and y values. It is
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

    def geoencode(self, **kwargs) -> alt.Chart:  # coverage: ignore
        """Returns an `altair <http://altair-viz.github.io/>`_ encoding of the
        shape to be composed in an interactive visualization.
        Specific plot features, such as line widths, can be passed via **kwargs.
        See `documentation
        <https://altair-viz.github.io/user_guide/marks.html>`_.
        """
        return (
            alt.Chart(
                self.data.query(
                    "latitude == latitude and longitude == longitude"
                )[["latitude", "longitude"]]
            )
            .encode(latitude="latitude", longitude="longitude")
            .mark_line(**kwargs)
        )


class GeoDBMixin(DataFrameMixin):
    def extent(
        self: T,
        extent: Union[str, ShapelyMixin, Tuple[float, float, float, float]],
        buffer: float = 0.5,
    ) -> Optional[T]:
        """
        Selects the subset of data inside the given extent.

        The parameter extent may be passed as:

            - a string to query OSM Nominatim service;
            - the result of an OSM Nominatim query;
            - any kind of shape (including airspaces);
            - extents (west, east, south, north)

        This works with databases like airways, airports or navaids.

        >>> airways.extent('Switzerland')

        >>> airports.extent(eurofirs['LFBB'])

        >>> navaids['ZUE']
        ZUE (NDB): 30.9 20.06833333 0 ZUEITINA NDB 369.0kHz
        >>> navaids.extent('Switzerland')['ZUE']
        ZUE (VOR): 47.59216667 8.81766667 1730 ZURICH EAST VOR-DME 110.05MHz

        """
        from ..drawing import Nominatim

        _extent = (0.0, 0.0, 0.0, 0.0)

        if isinstance(extent, str):
            _extent = Nominatim.search(extent).extent  # type: ignore
        if isinstance(extent, ShapelyMixin):
            _extent = extent.extent
        if isinstance(extent, Nominatim):
            _extent = extent.extent
        if isinstance(extent, tuple):
            _extent = extent

        west, east, south, north = _extent

        output = self.query(
            f"{south - buffer} <= latitude <= {north + buffer} and "
            f"{west - buffer} <= longitude <= {east + buffer}"
        )
        return output

    def geoencode(self, **kwargs) -> alt.Chart:  # coverage: ignore
        """Returns an `altair <http://altair-viz.github.io/>`_ encoding of the
        shape to be composed in an interactive visualization.
        """
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


class PointMixin(object):

    latitude: float
    longitude: float
    altitude: float
    timestamp: pd.Timestamp
    name: str

    @property
    def latlon(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    def plot(
        self, ax: Axes, text_kw=None, shift=None, **kwargs
    ) -> List[Artist]:  # coverage: ignore

        if shift is None:
            # flake B006
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
                text_kw["s"] = getattr(self, "callsign")  # noqa: B009
            if hasattr(self, "name"):
                text_kw["s"] = getattr(self, "name")  # noqa: B009

        cumul: List[Artist] = []
        cumul.append(ax.scatter(self.longitude, self.latitude, **kwargs))

        west, east, south, north = ax.get_extent(PlateCarree())
        if west <= self.longitude <= east and south <= self.latitude <= north:
            cumul.append(ax.text(self.longitude, self.latitude, **text_kw))

        return cumul


class _HBox(object):
    def __init__(self, *args):
        self.elts = args

    def _repr_html_(self):
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

    def __or__(self, other) -> "_HBox":
        if isinstance(other, _HBox):
            return _HBox(*self.elts, *other.elts)
        else:
            return _HBox(*self.elts, other)

    def __ror__(self, other) -> "_HBox":
        if isinstance(other, _HBox):
            return _HBox(*other.elts, *self.elts)
        else:
            return _HBox(other, *self.elts)


class HBoxMixin(object):
    """Enables a | operator for placing representations next to each other."""

    def __or__(self, other) -> _HBox:
        if isinstance(other, _HBox):
            return _HBox(self, *other.elts)
        else:
            return _HBox(self, other)
