from typing import Any

import plotly.express as px
import plotly.graph_objects as go


def line_map(
    self: Any,
    map_style: str = "carto-positron",
    **kwargs: Any,
) -> go.Figure:
    """Create a line plot with Plotly.

    Requires the plotly package (optional dependency).
    """
    if "center" not in kwargs:
        if (point := getattr(self, "point", None)) is not None:
            kwargs["center"] = point.latlon_dict

    return px.line_map(
        self.data,
        lat="latitude",
        lon="longitude",
        map_style=map_style,
        **kwargs,
    )


def scatter_map(
    self: Any, map_style: str = "carto-positron", **kwargs: Any
) -> go.Figure:
    """Create a scatter plot with Plotly.

    Requires the plotly package (optional dependency).
    """
    if "center" not in kwargs:
        if (point := getattr(self, "point", None)) is not None:
            kwargs["center"] = point.latlon_dict

    return px.scatter_map(
        self.data,
        lat="latitude",
        lon="longitude",
        map_style=map_style,
        **kwargs,
    )


def Scattermap(self: Any, **kwargs: Any) -> go.Scattermap:
    """Create a Scattermap with Plotly.

    Requires the plotly package (optional dependency).
    """
    return go.Scattermap(
        lat=self.data.latitude,
        lon=self.data.longitude,
        **kwargs,
    )


def line_geo(self: Any, **kwargs: Any) -> go.Figure:
    """Create a line plot with Plotly.

    Requires the plotly package (optional dependency).
    """
    if "center" not in kwargs:
        if (point := getattr(self, "point", None)) is not None:
            kwargs["center"] = point.latlon_dict

    return px.line_geo(
        self.data,
        lat="latitude",
        lon="longitude",
        **kwargs,
    )


def scatter_geo(self: Any, **kwargs: Any) -> go.Figure:
    """Create a scatter plot with Plotly.

    Requires the plotly package (optional dependency).
    """
    if "center" not in kwargs:
        if (point := getattr(self, "point", None)) is not None:
            kwargs["center"] = point.latlon_dict

    return px.scatter_geo(
        self.data,
        lat="latitude",
        lon="longitude",
        **kwargs,
    )


def Scattergeo(self: Any, **kwargs: Any) -> go.Scattergeo:
    """Create a Scattergeo with Plotly.

    Requires the plotly package (optional dependency).
    """
    return go.Scattergeo(
        lat=self.data.latitude,
        lon=self.data.longitude,
        **kwargs,
    )
