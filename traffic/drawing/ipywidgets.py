# fmt: off

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import clear_output
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from ipywidgets import (Button, Dropdown, HBox, Output, SelectionRangeSlider,
                        SelectMultiple, Tab, Text, VBox)

from . import *  # noqa: F401, F403, type: ignore
from ..core import Traffic
from ..drawing import (EuroPP, PlateCarree, Projection,  # type: ignore
                       countries, location, rivers)

# fmt: on


class TrafficWidget(object):

    # -- Constructor --
    def __init__(self, traffic: Traffic, projection=EuroPP()) -> None:

        ipython = get_ipython()
        ipython.magic("matplotlib ipympl")
        from ipympl.backend_nbagg import FigureCanvasNbAgg, FigureManagerNbAgg

        self.fig_map = Figure(figsize=(6, 6))
        self.fig_time = Figure(figsize=(6, 4))

        self.canvas_map = FigureCanvasNbAgg(self.fig_map)
        self.canvas_time = FigureCanvasNbAgg(self.fig_time)

        self.manager_map = FigureManagerNbAgg(self.canvas_map, 0)
        self.manager_time = FigureManagerNbAgg(self.canvas_time, 0)

        layout = {"width": "590px", "height": "800px", "border": "none"}
        self.output = Output(layout=layout)

        self._traffic = traffic
        self.t_view = traffic.sort_values("timestamp")
        self.trajectories: Dict[str, List[Artist]] = defaultdict(list)

        self.create_map(projection)

        self.projection = Dropdown(options=["EuroPP", "Lambert93", "Mercator"])
        self.projection.observe(self.on_projection_change)

        self.identifier_input = Text(description="Callsign/ID")
        self.identifier_input.observe(self.on_id_input)

        self.identifier_select = SelectMultiple(
            options=sorted(self._traffic.callsigns),  # type: ignore
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

        self.time_slider = SelectionRangeSlider(
            options=list(range(100)),
            index=(0, 99),
            description="Date",
            continuous_update=False,
        )
        self.lock_time_change = False
        self.set_time_range()

        self.time_slider.observe(self.on_time_select)
        self.canvas_map.observe(
            self.on_axmap_change, ["_button", "_png_is_old"]
        )
        self.canvas_time.observe(self.on_axtime_change, ["_png_is_old"])

        self.tabs = Tab()
        self.tabs.children = [self.canvas_map, self.canvas_time]
        self.tabs.set_title(0, "Map")
        self.tabs.set_title(1, "Plots")

        self._main_elt = HBox(
            [
                self.tabs,
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

    @property
    def traffic(self) -> Traffic:
        return self._traffic

    def _ipython_display_(self) -> None:
        clear_output()
        self.canvas_map.draw_idle()
        self._main_elt._ipython_display_()

    def debug(self) -> None:
        if self.tabs.children[-1] != self.output:
            self.tabs.children = list(self.tabs.children) + [self.output]

    def set_time_range(self) -> None:
        with self.output:
            tz_now = datetime.now().astimezone().tzinfo
            self.dates = [
                self._traffic.start_time
                + i * (self._traffic.end_time - self._traffic.start_time) / 99
                for i in range(100)
            ]
            if self._traffic.start_time.tzinfo is not None:
                options = [
                    t.tz_convert("utc").strftime("%H:%M") for t in self.dates
                ]
            else:
                options = [
                    t.tz_localize(tz_now).tz_convert("utc").strftime("%H:%M")
                    for t in self.dates
                ]

            self.lock_time_change = True
            self.time_slider.options = options
            self.time_slider.index = (0, 99)
            self.lock_time_change = False

    def create_map(
        self, projection: Union[str, Projection] = "EuroPP()"  # type: ignore
    ) -> None:
        with self.output:
            if isinstance(projection, str):
                if not projection.endswith("()"):
                    projection = projection + "()"
                projection = eval(projection)

            self.projection = projection

            with plt.style.context("traffic"):

                self.fig_map.clear()
                self.trajectories.clear()
                self.ax_map = self.fig_map.add_subplot(
                    111, projection=self.projection
                )
                self.ax_map.add_feature(countries())
                if projection.__class__.__name__.split(".")[-1] in [
                    "Lambert93"
                ]:
                    self.ax_map.add_feature(rivers())

                self.fig_map.set_tight_layout(True)
                self.ax_map.background_patch.set_visible(False)
                self.ax_map.outline_patch.set_visible(False)
                self.ax_map.format_coord = lambda x, y: ""
                self.ax_map.set_global()

            self.default_plot()
            self.canvas_map.draw_idle()

    def default_plot(self) -> None:
        with self.output:
            # clear all trajectory pieces
            for key, value in self.trajectories.items():
                for elt in value:
                    elt.remove()
            self.trajectories.clear()
            self.ax_map.set_prop_cycle(None)

            lon_min, lon_max, lat_min, lat_max = self.ax_map.get_extent(
                PlateCarree()
            )
            cur_flights = list(
                f.at()
                for f in self.t_view
                if lat_min <= getattr(f.at(), "latitude", -90) <= lat_max
                and lon_min <= getattr(f.at(), "longitude", -180) <= lon_max
            )

            def params(at):
                if len(cur_flights) < 10:
                    return dict(s=8, text_kw=dict(s=at.callsign))
                else:
                    return dict(s=8, text_kw=dict(s=""))

            for at in cur_flights:
                self.trajectories[at.callsign] += at.plot(self.ax_map, **params(at))

            self.canvas_map.draw_idle()

    def create_timeplot(self) -> None:
        with plt.style.context("traffic"):
            self.fig_time.clear()
            self.ax_time = self.fig_time.add_subplot(111)
            self.fig_time.set_tight_layout(True)

    # -- Callbacks --

    def on_projection_change(self, change: Dict[str, Any]) -> None:
        with self.output:
            if change["name"] == "value":
                self.create_map(change["new"])

    def on_clear_button(self, elt: Dict[str, Any]) -> None:
        with self.output:
            self.t_view = self.traffic.sort_values("timestamp")
            self.create_map(self.projection)
            self.create_timeplot()

    def on_area_input(self, elt: Dict[str, Any]) -> None:
        with self.output:
            if elt["name"] != "value":
                return
            search_text = elt["new"]
            if len(search_text) == 0:
                self.area_select.options = list()
            else:
                from ..data import airac

                self.area_select.options = list(
                    x.name for x in airac.parse(search_text)
                )

    def on_area_click(self, elt: Dict[str, Any]) -> None:
        with self.output:
            if elt["name"] != "value":
                return
            from ..data import airac

            self.ax_map.set_extent(airac[elt["new"][0]])
            self.canvas_map.draw_idle()

    def on_extent_button(self, elt: Dict[str, Any]) -> None:
        with self.output:
            if len(self.area_select.value) == 0:
                if len(self.area_input.value) == 0:
                    self.ax_map.set_global()
                else:
                    self.ax_map.set_extent(location(self.area_input.value))
            else:
                from ..data import airac

                self.ax_map.set_extent(airac[self.area_select.value[0]])

            t1, t2 = self.time_slider.index
            low, up = self.altitude_select.value
            self.on_filter(low, up, t1, t2)
            self.canvas_map.draw_idle()

    def on_axtime_change(self, change: Dict[str, Any]) -> None:
        with self.output:
            if change["name"] == "_png_is_old":
                # go away!!
                return self.canvas_map.set_window_title("")

    def on_axmap_change(self, change: Dict[str, Any]) -> None:
        with self.output:
            if change["name"] == "_png_is_old":
                # go away!!
                return self.canvas_map.set_window_title("")
            if change["new"] is None:
                t1, t2 = self.time_slider.index
                low, up = self.altitude_select.value
                self.on_filter(low, up, t1, t2)

    def on_id_input(self, elt: Dict[str, Any]) -> None:
        with self.output:
            # low, up = alt.value
            self.identifier_select.options = sorted(
                callsign
                for callsign in self.t_view.callsigns
                if re.match(elt["new"]["value"], callsign, flags=re.IGNORECASE)
            )

    def on_plot_button(self, elt: Dict[str, Any]) -> None:
        with self.output:
            if len(self.area_select.value) == 0:
                if len(self.area_input.value) == 0:
                    return self.default_plot()
                location(self.area_input.value).plot(
                    self.ax_map, color="grey", linestyle="dashed"
                )
            else:
                from ..data import airac

                airspace = airac[self.area_select.value[0]]
                if airspace is not None:
                    airspace.plot(self.ax_map)
            self.canvas_map.draw_idle()

    def on_plot_airport(self, elt: Dict[str, Any]) -> None:
        with self.output:
            if len(self.area_input.value) == 0:
                from cartotools.osm import request, tags

                west, east, south, north = self.ax_map.get_extent(crs=PlateCarree())
                if abs(east - west) > 1 or abs(north - south) > 1:
                    # that would be a too big request
                    return
                request((west, south, east, north), **tags.airport).plot(
                    self.ax_map
                )
            else:
                from ..data import airports

                airports[self.area_input.value].plot(self.ax_map)
            self.canvas_map.draw_idle()

    def on_id_change(self, change: Dict[str, Any]) -> None:
        with self.output:
            if change["name"] != "value":
                return

            # clear all trajectory pieces
            self.create_timeplot()
            for key, value in self.trajectories.items():
                for elt in value:
                    elt.remove()
            self.trajectories.clear()

            callsigns = change["new"]
            for c in callsigns:
                f = self.t_view[c]
                if f is not None:
                    try:
                        self.trajectories[c] += f.plot(self.ax_map)
                        self.trajectories[c] += f.at().plot(
                            self.ax_map, s=8, text_kw=dict(s=c)
                        )
                    except TypeError:  # NoneType object is not iterable
                        pass

                    try:
                        f.plot_time(
                            self.ax_time, y=["altitude"], label=f.callsign
                        )
                    except TypeError:  # no numeric data to plot
                        pass

            if len(callsigns) == 0:
                self.default_plot()
            else:
                self.ax_time.legend()

            # non conformal with traffic style
            for elt in self.ax_time.get_xticklabels():
                elt.set_size(12)
            for elt in self.ax_time.get_yticklabels():
                elt.set_size(12)
            self.ax_time.set_xlabel("")

            self.canvas_map.draw_idle()
            self.canvas_time.draw_idle()

            if len(callsigns) != 0:
                low, up = self.ax_time.get_ylim()
                if (up - low) / up < 0.05:
                    self.ax_time.set_ylim(up - .05 * up, up + .05 * up)
                    self.canvas_time.draw_idle()

    def on_filter(self, low, up, t1, t2) -> None:
        with self.output:
            west, east, south, north = self.ax_map.get_extent(crs=PlateCarree())

            self.t_view = (
                self.traffic.between(self.dates[t1], self.dates[t2])
                .query(f"{low} <= altitude <= {up} or altitude != altitude")
                .query(
                    f"{west} <= longitude <= {east} and "
                    f"{south} <= latitude <= {north}"
                )
                .sort_values("timestamp")
            )
            self.identifier_select.options = sorted(
                flight.callsign
                for flight in self.t_view
                if flight is not None
                and re.match(
                    self.identifier_input.value,
                    flight.callsign,
                    flags=re.IGNORECASE,
                )
            )
            return self.default_plot()

    def on_altitude_select(self, change: Dict[str, Any]) -> None:
        with self.output:
            if change["name"] != "value":
                return

            low, up = change["new"]
            t1, t2 = self.time_slider.index
            self.on_filter(low, up, t1, t2)

    def on_time_select(self, change: Dict[str, Any]) -> None:
        with self.output:
            if self.lock_time_change:
                return
            if change["name"] != "index":
                return
            t1, t2 = change["new"]
            low, up = self.altitude_select.value
            self.on_filter(low, up, t1, t2)
