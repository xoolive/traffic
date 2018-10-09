# fmt: off

import logging
import sys
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Union

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy

from cartopy.crs import PlateCarree, Projection

from ..core import Traffic
from ..drawing import *  # noqa: F401, F403, type: ignore
from ..drawing import countries, rivers

# fmt: on


class NavigationToolbar(NavigationToolbar2QT):
    """Emulates a toolbar but do not display it."""

    def set_message(self, msg):
        pass


class TimeCanvas(FigureCanvasQTAgg):
    """Plotting info with UTC timestamp on the x-axis."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        logging.info("Initialize TimeCanvas")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.main = parent
        self.trajectories: Dict[str, List[Artist]] = defaultdict(list)

        self.lock = Lock()

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self)

        self.create_plot()

    def create_plot(self):
        with plt.style.context("traffic"):
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.fig.set_tight_layout(True)

    def plot_callsigns(
        self,
        traffic: Traffic,
        callsigns: List[str],
        y: List[str],
        secondary_y: List[str],
    ) -> None:

        if len(y) == 0:
            y = ["altitude"]

        extra_dict = dict()

        if len(y) > 1:
            # just to avoid confusion...
            callsigns = callsigns[:1]

        for key, value in self.trajectories.items():
            for elt in value:
                elt.remove()
        self.trajectories.clear()

        for callsign in callsigns:
            flight = traffic[callsign]
            if len(y) == 1:
                extra_dict["label"] = callsign
            if flight is not None:
                try:
                    flight.plot_time(
                        self.ax, y=y, secondary_y=secondary_y, **extra_dict
                    )
                except Exception:  # no numeric data to plot
                    pass

        if len(callsigns) > 1:
            self.ax.legend()

        for elt in self.ax.get_xticklabels():
            elt.set_size(12)
        for elt in self.ax.get_yticklabels():
            elt.set_size(12)
        self.ax.set_xlabel("")

        if len(callsigns) > 0:
            low, up = self.ax.get_ylim()
            if (up - low) / up < 0.05:
                self.ax.set_ylim(up - .05 * up, up + .05 * up)

        if len(callsigns) > 0 and len(secondary_y) > 0:
            ax2, _ = next(iter(self.ax.get_shared_x_axes()))
            low, up = ax2.get_ylim()
            if (up - low) / up < 0.05:
                ax2.set_ylim(up - .05 * up, up + .05 * up)

        self.draw()

    def draw(self):
        with self.lock:
            if self.fig is None:
                return
            super().draw()


class MapCanvas(FigureCanvasQTAgg):
    """Plotting maps."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        logging.info("Initialize MapCanvas")
        self.trajectories = defaultdict(list)

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.main = parent
        self.lock = Lock()

        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self)
        self.create_map()

    def wheelEvent(self, event):
        if sys.platform == "darwin":  # rather use pinch
            return
        self.zoom(event.angleDelta().y() > 0, 0.8)

    def zoom(self, zoom_in, factor):
        min_x, max_x, min_y, max_y = self.ax.axis()
        if not zoom_in:
            factor = 1.0 / factor

        center_x = .5 * (max_x + min_x)
        delta_x = .5 * (max_x - min_x)
        center_y = .5 * (max_y + min_y)
        delta_y = .5 * (max_y - min_y)

        self.ax.axis(
            (
                center_x - factor * delta_x,
                center_x + factor * delta_x,
                center_y - factor * delta_y,
                center_y + factor * delta_y,
            )
        )

        self.fig.tight_layout()
        self.lims = self.ax.axis()
        fmt = ", ".join("{:.5e}".format(t) for t in self.lims)
        logging.info("Zooming to {}".format(fmt))
        self.draw()

    def create_map(
        self, projection: Union[str, Projection] = "EuroPP()"  # type: ignore
    ) -> None:
        if isinstance(projection, str):
            if not projection.endswith(")"):
                projection = projection + "()"
            projection = eval(projection)

        self.projection = projection
        self.trajectories.clear()

        with plt.style.context("traffic"):

            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection=self.projection)
            projection_name = projection.__class__.__name__.split(".")[-1]

            self.ax.add_feature(
                countries(
                    scale="10m"
                    if projection_name not in ["Mercator", "Orthographic"]
                    else "110m"
                )
            )
            if projection_name in ["Lambert93", "GaussKruger", "Amersfoort"]:
                self.ax.add_feature(rivers())

            self.fig.set_tight_layout(True)
            self.ax.background_patch.set_visible(False)
            self.ax.outline_patch.set_visible(False)
            self.ax.format_coord = lambda x, y: ""
            self.ax.set_global()

        self.draw()

    def plot_callsigns(self, traffic: Traffic, callsigns: List[str]) -> None:
        if traffic is None:
            return

        for key, value in self.trajectories.items():
            for elt in value:
                elt.remove()
        self.trajectories.clear()
        self.ax.set_prop_cycle(None)

        for c in callsigns:
            f = traffic[c]
            if f is not None:
                try:
                    self.trajectories[c] += f.plot(self.ax)
                    f_at = f.at()
                    if (
                        f_at is not None
                        and hasattr(f_at, "latitude")
                        and f_at.latitude == f_at.latitude
                    ):
                        self.trajectories[c] += f_at.plot(
                            self.ax, s=8, text_kw=dict(s=c)
                        )
                except TypeError:  # NoneType object is not iterable
                    pass

        if len(callsigns) == 0:
            self.default_plot(traffic)

        self.draw()

    def default_plot(self, traffic: Traffic) -> None:
        if traffic is None:
            return
        # clear all trajectory pieces
        for key, value in self.trajectories.items():
            for elt in value:
                elt.remove()
        self.trajectories.clear()

        lon_min, lon_max, lat_min, lat_max = self.ax.get_extent(PlateCarree())
        cur_ats = list(f.at() for f in traffic)
        cur_flights = list(
            at
            for at in cur_ats
            if at is not None
            if hasattr(at, "latitude")
            and at.latitude is not None
            and lat_min <= at.latitude <= lat_max
            and lon_min <= at.longitude <= lon_max
        )

        def params(at):
            if len(cur_flights) < 10:
                return dict(s=8, text_kw=dict(s=at.callsign))
            else:
                return dict(s=8, text_kw=dict(s=""))

        for at in cur_flights:
            if at is not None:
                self.trajectories[at.callsign] += at.plot(self.ax, **params(at))

        self.draw()

    def draw(self):
        with self.lock:
            if self.fig is None:
                return
            super().draw()
