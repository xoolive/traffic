from typing import Any

import plotly.express as px
import plotly.graph_objects as go


def line_mapbox(
    self: Any,
    mapbox_style: str = "carto-positron",
    **kwargs: Any,
) -> go.Figure:
    """Create a line plot with Plotly.

    Requires the plotly package (optional dependency).
    """
    if "center" not in kwargs:
        if (point := getattr(self, "point", None)) is not None:
            kwargs["center"] = point.latlon_dict

    return px.line_mapbox(
        self.data,
        lat="latitude",
        lon="longitude",
        mapbox_style=mapbox_style,
        **kwargs,
    )


def scatter_mapbox(
    self: Any, mapbox_style: str = "carto-positron", **kwargs: Any
) -> go.Figure:
    """Create a scatter plot with Plotly.

    Requires the plotly package (optional dependency).
    """
    if "center" not in kwargs:
        if (point := getattr(self, "point", None)) is not None:
            kwargs["center"] = point.latlon_dict

    return px.scatter_mapbox(
        self.data,
        lat="latitude",
        lon="longitude",
        mapbox_style=mapbox_style,
        **kwargs,
    )


def Scattermapbox(self: Any, **kwargs: Any) -> go.Scattermapbox:
    """Create a Scattermapbox with Plotly.

    Requires the plotly package (optional dependency).
    """
    return go.Scattermapbox(
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
    """Create a Scattermapbox with Plotly.

    Requires the plotly package (optional dependency).
    """
    return go.Scattergeo(
        lat=self.data.latitude,
        lon=self.data.longitude,
        **kwargs,
    )
