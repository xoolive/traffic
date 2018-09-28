# fmt: off

import re
from datetime import datetime
from typing import Any, Dict, Union

from IPython import get_ipython

from ipywidgets import (Button, Dropdown, HBox, SelectionRangeSlider,
                        SelectMultiple, Text, VBox)

from . import *  # noqa: F401, F403, type: ignore
from ..core import Traffic
from ..data import airac, airports
from ..drawing import (EuroPP, PlateCarree, Projection,  # type: ignore
                       countries, location, rivers)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# fmt: on


class TrafficWidget(object):
    def __init__(self, traffic: Traffic, projection=EuroPP()) -> None:
        ipython = get_ipython()
        ipython.magic("matplotlib ipympl")
        from ipympl.backend_nbagg import FigureCanvasNbAgg, FigureManagerNbAgg

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasNbAgg(self.fig)
        self.manager = FigureManagerNbAgg(self.canvas, 0)

        self.traffic = traffic
        self.create_map(projection)

        self.trajectories: Dict[str, Any] = dict()
        self.points: Dict[str, Any] = dict()

        self.projection = Dropdown(options=["EuroPP", "Lambert93", "Mercator"])
        self.projection.observe(self.on_projection_change)

        self.identifier_input = Text(description="Callsign/ID")
        self.identifier_input.observe(self.on_id_input)

        self.identifier_select = SelectMultiple(
            options=sorted(self.traffic.callsigns),  # type: ignore
            value=[],
            rows=20,
        )
        self.identifier_select.observe(self.on_id_change)

        self.area_input = Text(description="Area")
        self.area_input.observe(self.on_area_input)

        self.extent_button = Button(description="Extent")
        self.extent_button.on_click(self.on_extent_button)

        self.plot_button = Button(description="Plot")
        self.plot_button.on_click(self.on_plot_button)

        self.clear_button = Button(description="Reset")
        self.clear_button.on_click(self.on_clear_button)

        self.plot_airport = Button(description="Airport")
        self.plot_airport.on_click(self.on_plot_airport)

        self.area_select = SelectMultiple(
            options=[], value=[], rows=3, disabled=False
        )
        self.area_select.observe(self.on_area_click)

        self.altitude_select = SelectionRangeSlider(
            options=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
            index=(0, 8),
            description="Altitude",
            disabled=False,
            continuous_update=False,
        )
        self.altitude_select.observe(self.on_altitude_select)

        tz_now = datetime.now().astimezone().tzinfo
        self.dates = [
            self.traffic.start_time
            + i * (self.traffic.end_time - self.traffic.start_time) / 99
            for i in range(100)
        ]
        if self.traffic.start_time.tzinfo is not None:
            options = [
                t.tz_convert("utc").strftime("%H:%M") for t in self.dates
            ]
        else:
            options = [
                t.tz_localize(tz_now).tz_convert("utc").strftime("%H:%M")
                for t in self.dates
            ]

        self.time_slider = SelectionRangeSlider(
            options=options,
            index=(0, 99),
            description="Date",
            continuous_update=False,
        )
        self.time_slider.observe(self.on_time_select)

        self._main_elt = HBox(
            [
                self.canvas,
                VBox(
                    [
                        self.projection,
                        HBox([self.extent_button, self.plot_button]),
                        HBox([self.plot_airport, self.clear_button]),
                        self.area_input,
                        self.area_select,
                        self.time_slider,
                        self.altitude_select,
                        self.identifier_input,
                        self.identifier_select,
                    ]
                ),
            ]
        )

    def _ipython_display_(self):
        self.canvas.draw_idle()
        self._main_elt._ipython_display_()
        self.canvas.set_window_title("")

    def create_map(
        self, projection: Union[str, Projection] = "EuroPP()"  # type: ignore
    ):
        if isinstance(projection, str):
            if not projection.endswith("()"):
                projection = projection + "()"
            projection = eval(projection)

        self.projection = projection

        with plt.style.context("traffic"):

            self.fig.clear()
            self.ax = self.fig.add_subplot(111, projection=self.projection)
            self.ax.add_feature(countries())
            if projection.__class__.__name__.split(".")[-1] in [
                # "EuroPP",
                "Lambert93"
            ]:
                self.ax.add_feature(rivers())

            self.fig.set_tight_layout(True)
            self.ax.background_patch.set_visible(False)
            self.ax.outline_patch.set_visible(False)
            self.ax.format_coord = lambda x, y: ""
            self.ax.set_global()

        self.canvas.draw_idle()

    def on_projection_change(self, elt):
        self._debug = elt
        projection_idx = elt["new"]["index"]
        self.create_map(elt["owner"].options[projection_idx])

    def on_clear_button(self, elt):
        self.create_map(self.projection)

    def on_area_input(self, elt):
        search_text = elt["new"]["value"]
        if len(search_text) == 0:
            self.area_select.options = list()
        else:
            self.area_select.options = list(
                x.name for x in airac.parse(search_text)
            )

    def on_area_click(self, elt):
        if elt["name"] != "value":
            return
        self.ax.set_extent(airac[elt["new"][0]])
        self.canvas.draw_idle()

    def on_extent_button(self, elt):
        if len(self.area_select.value) == 0:
            if len(self.area_input.value) == 0:
                self.ax.set_global()
            else:
                self.ax.set_extent(location(self.area_input.value))
        else:
            self.ax.set_extent(airac[self.area_select.value[0]])
        self.canvas.draw_idle()

    def on_id_input(self, elt):
        # low, up = alt.value
        self.identifier_select.options = sorted(
            c
            for c in self.traffic.callsigns
            if re.match(elt["new"]["value"], c, flags=re.IGNORECASE)
        )

    def on_plot_button(self, elt):
        if len(self.area_select.value) == 0:
            location(self.area_input.value).plot(
                self.ax, color="grey", linestyle="dashed"
            )
        else:
            airac[self.area_select.value[0]].plot(self.ax, color="crimson")
        self.canvas.draw_idle()

    def on_plot_airport(self, elt):
        if len(self.area_input.value) == 0:
            from cartotools.osm import request, tags

            west, east, south, north = self.ax.get_extent(crs=PlateCarree())
            if abs(east - west) > 1 or abs(north - south) > 1:
                # that would be a too big request
                return
            request((west, south, east, north), **tags.airport).plot(self.ax)
        else:
            airports[self.area_input.value].plot(self.ax)
        self.canvas.draw_idle()

    def on_id_change(self, change):
        if change["name"] != "value":
            return
        for c in change["old"]:
            if c in self.trajectories:
                self.trajectories[c].set_visible(False)
        callsigns = change["new"]
        for c in callsigns:
            if c in self.trajectories:
                self.trajectories[c].set_visible(
                    not self.trajectories[c].get_visible()
                )
            else:
                f = self.traffic[c]
                if f is not None:
                    self.trajectories[c], *_ = f.plot(self.ax)
        self.canvas.draw_idle()

    def on_altitude_select(self, change):
        low, up = change["new"]["index"]
        t1, t2 = self.time_slider.index
        self.identifier_select.options = sorted(
            flight.callsign
            for flight in self.traffic.between(self.dates[t1], self.dates[t2])
            if flight is not None
            if re.match(
                self.identifier_input.value,
                flight.callsign,
                flags=re.IGNORECASE,
            )
            and flight.data.altitude.max() > change["owner"].options[low]
            and flight.data.altitude.min() < change["owner"].options[up]
        )

    def on_time_select(self, change):
        t1, t2 = change["new"]["index"]
        low, up = self.altitude_select.value
        self.identifier_select.options = sorted(
            flight.callsign
            for flight in self.traffic.between(self.dates[t1], self.dates[t2])
            if flight is not None
            if re.match(
                self.identifier_input.value,
                flight.callsign,
                flags=re.IGNORECASE,
            )
            and flight.data.altitude.max() > low
            and flight.data.altitude.min() < up
        )
