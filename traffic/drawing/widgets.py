# fmt: off

import re
from datetime import datetime

import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import clear_output

from ipywidgets import (Button, HBox, SelectionRangeSlider, SelectMultiple,
                        Text, VBox)

from .cartopy import EuroPP, countries, location, rivers  # noqa
from ..core import Traffic
from ..data import airac

# fmt: on


class TrafficWidget(object):
    def __init__(self, traffic: Traffic) -> None:
        ipython = get_ipython()
        ipython.magic("matplotlib ipympl")

        self.traffic = traffic

        with plt.style.context("traffic"):
            self.ax = plt.axes(projection=EuroPP())
            self.ax.clear()
            self.ax.add_feature(countries())
            self.ax.add_feature(rivers())

            self.ax.figure.set_tight_layout(True)
            self.ax.figure.set_size_inches((6, 6))
            self.ax.background_patch.set_visible(False)
            self.ax.outline_patch.set_visible(False)
            self.ax.format_coord = lambda x, y: ""
            self.ax.set_global()

        self.trajectories = dict()
        self.points = dict()

        self.identifier_input = Text()
        self.identifier_input.observe(self.on_id_input)

        self.identifier_select = SelectMultiple(
            options=sorted(self.traffic.callsigns), value=[], rows=20
        )
        self.identifier_select.observe(self.on_id_change)

        self.area_input = Text()
        self.area_input.observe(self.on_area_input)

        self.extent_button = Button(description="Extent")
        self.extent_button.on_click(self.on_extent_button)

        self.plot_button = Button(description="Plot")
        self.plot_button.on_click(self.on_plot_button)

        self.area_select = SelectMultiple(
            options=[], value=[], rows=3, disabled=False
        )
        self.area_select.observe(self.on_area_click)

        self.altitude_select = SelectionRangeSlider(
            options=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
            index=(0, 8),
            description="altitude",
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
            description="date",
            continuous_update=False,
        )
        self.time_slider.observe(self.on_time_select)

        self._main_elt = HBox(
            [
                self.ax.figure.canvas,
                VBox(
                    [
                        self.area_input,
                        HBox([self.extent_button, self.plot_button]),
                        self.area_select,
                        self.identifier_input,
                        self.time_slider,
                        self.altitude_select,
                        self.identifier_select,
                    ]
                ),
            ]
        )

    def _ipython_display_(self):
        clear_output()
        return self._main_elt._ipython_display_()

    def on_area_input(self, elt):
        search_text = elt["new"]["value"]
        if len(search_text) > 0:
            self.area_select.options = list(
                x.name for x in airac.parse(search_text)
            )

    def on_area_click(self, elt):
        if elt["name"] != "value":
            return
        self.ax.set_extent(airac[elt["new"][0]])

    def on_extent_button(self, elt):
        if len(self.area_select.value) == 0:
            if len(self.area_input.value) == 0:
                self.ax.set_global()
            else:
                self.ax.set_extent(location(self.area_input.value))
        else:
            self.ax.set_extent(airac[self.area_select.value[0]])

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
