# fmt: off

import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, cast

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QGridLayout,
                             QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                             QListWidget, QMainWindow, QMessageBox,
                             QPushButton, QSlider, QSystemTrayIcon, QTabWidget,
                             QVBoxLayout, QWidget)

from cartopy.crs import PlateCarree

from .. import config
from ..core import Traffic, Flight
from ..data import ModeS_Decoder, airac
from ..drawing import location
from .plot import MapCanvas, NavigationToolbar2QT, TimeCanvas

# fmt: on


def dont_crash(fn):
    """
    Wraps callbacks: a simple information is raised in place of a program crash.
    """

    def safe_exec(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            logging.exception(e)
            QMessageBox.information(
                self, type(e).__name__, " ".join(str(x) for x in e.args)
            )

    return safe_exec


class UpdateTraffic(QtCore.QThread):
    """
    UpdateTraffic periodically checks the content of a decoded traffic.
    """

    def __init__(self, parent: "MainScreen", refresh_time: int) -> None:
        super().__init__()
        self.parent = parent
        self.refresh_time = refresh_time

    def run(self):
        while True:
            delta = datetime.now() - self.parent.last_interact
            # do not overreact, stay minimalist!!
            if delta.total_seconds() > self.refresh_time:
                self.parent._tview = self.parent.traffic
                if self.parent._tview is not None:
                    self.parent.map_plot.default_plot(self.parent._tview)
                    self.parent.map_plot.draw()
            time.sleep(1)

    def __del__(self):
        self.thread.quit()
        while self.thread.isRunning():
            time.sleep(1)


class AiracInitCache(QtCore.QThread):
    """Initialize cache in background to avoid lag.
    """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):
        try:
            airac.init_cache()
            self.parent.airac_ready = True
        except Exception:
            pass


class MainScreen(QMainWindow):
    """The Main GUI layout and callbacks.
    """

    def __init__(self) -> None:

        logging.info("Initialize MainScreen")
        super().__init__(parent=None)

        self._traffic: Optional[Traffic] = None
        self._tview: Optional[Traffic] = None
        self.decoder: Optional[ModeS_Decoder] = None
        self.updateTraffic: Optional[UpdateTraffic] = None
        self.airac_ready: bool = False
        self.last_interact: datetime = datetime.now()

        airac_init = AiracInitCache(self)
        airac_init.start()

        self.setWindowTitle("traffic")
        self.setGeometry(10, 10, 920, 720)

        self.set_icons()
        self.set_layout()
        self.set_design()
        self.set_callbacks()

    def __del__(self) -> None:
        if self.updateTraffic is not None:
            self.updateTraffic.terminate()
        if self.decoder is not None:
            self.decoder.stop()

    @property
    def traffic(self) -> Optional[Traffic]:
        self.last_interact = datetime.now()
        if self.decoder is not None:
            self._traffic = self.decoder.traffic
            if self._traffic is None:
                return None
            self.set_time_range()
            # self.set_float_columns()
        #            max_alt = 100 * self.altitude_slider.value()
        #            max_time = self.dates[self.time_slider.value()]
        #            self.on_filter(max_alt, max_time)
        return self._traffic

    def set_callbacks(self) -> None:
        self.airport_button.clicked.connect(self.on_plot_airport)
        self.altitude_slider.sliderMoved.connect(self.on_altitude_moved)
        self.altitude_slider.sliderReleased.connect(self.on_select)
        self.area_input.textEdited.connect(self.on_area_input)
        self.area_select.itemSelectionChanged.connect(self.on_area_select)
        self.extent_button.clicked.connect(self.on_extent_button)
        self.identifier_input.textEdited.connect(self.on_id_input)
        self.identifier_select.itemSelectionChanged.connect(self.on_id_change)
        self.open_dropdown.activated.connect(self.on_open)
        self.plot_button.clicked.connect(self.on_plot_button)
        self.projection_dropdown.activated.connect(self.make_map)
        self.reset_button.clicked.connect(self.on_clear_button)
        self.time_slider.sliderMoved.connect(self.on_time_moved)
        self.time_slider.sliderReleased.connect(self.on_select)
        self.y_selector.itemSelectionChanged.connect(self.on_id_change)
        self.sec_y_selector.itemSelectionChanged.connect(self.on_id_change)

    def set_design(self) -> None:
        self.open_dropdown.setMaximumWidth(400)
        self.projection_dropdown.setMaximumWidth(400)
        self.altitude_description.setMinimumWidth(100)
        self.altitude_description.setMaximumWidth(100)
        self.altitude_slider_info.setMinimumWidth(50)
        self.altitude_slider_info.setMaximumWidth(50)
        self.altitude_slider.setMaximumWidth(240)
        self.area_input_description.setMinimumWidth(100)
        self.area_input_description.setMaximumWidth(100)
        self.area_input.setMaximumWidth(290)
        self.area_select.setMaximumHeight(100)
        self.area_select.setMaximumWidth(400)
        self.identifier_description.setMinimumWidth(100)
        self.identifier_description.setMaximumWidth(100)
        self.identifier_input.setMaximumWidth(290)
        self.identifier_select.setMaximumWidth(400)
        self.time_description.setMinimumWidth(100)
        self.time_description.setMaximumWidth(100)
        self.time_slider_info.setMinimumWidth(50)
        self.time_slider_info.setMaximumWidth(50)
        self.time_slider.setMaximumWidth(240)
        self.y_selector.setMaximumHeight(100)
        self.sec_y_selector.setMaximumHeight(100)

    def set_layout(self) -> None:

        self.plot_tabs = QTabWidget()
        map_tab = QWidget()
        map_layout = QVBoxLayout()
        map_tab.setLayout(map_layout)

        self.map_plot = MapCanvas(parent=self, width=5, height=4)
        self.map_plot.move(0, 0)
        self.time_plot = TimeCanvas(parent=self, width=5, height=4)
        self.time_plot.move(0, 0)

        map_toolbar = NavigationToolbar2QT(self.map_plot, map_tab)
        map_toolbar.setVisible(False)
        map_layout.addWidget(map_toolbar)
        map_layout.addWidget(self.map_plot)
        map_toolbar.pan()

        time_tab = QWidget()
        time_layout = QVBoxLayout()
        time_tab.setLayout(time_layout)

        self.y_selector = QListWidget()
        self.sec_y_selector = QListWidget()
        self.y_selector.setSelectionMode(3)  # extended selection
        self.sec_y_selector.setSelectionMode(3)  # extended selection
        selector = QHBoxLayout()
        selector.addWidget(self.y_selector)
        selector.addWidget(self.sec_y_selector)

        time_layout.addLayout(selector)
        time_layout.addWidget(self.time_plot)

        self.plot_tabs.addTab(map_tab, "Map")
        self.plot_tabs.addTab(time_tab, "Plots")

        plot_column = QVBoxLayout()
        plot_column.addWidget(self.plot_tabs)

        self.interact_column = QVBoxLayout()

        self.open_options = ["Open file", "dump1090"]
        if "decoders" in config:
            self.open_options += list(config["decoders"])
        self.open_dropdown = QComboBox()
        for option in self.open_options:
            self.open_dropdown.addItem(option)
        self.interact_column.addWidget(self.open_dropdown)

        self.projections = ["EuroPP", "Lambert93", "Mercator"]
        self.projection_dropdown = QComboBox()
        more_projs = config.get("projections", "extra", fallback="")
        if more_projs != "":
            proj_list = more_projs.split(";")
            self.projections += list(x.strip() for x in proj_list)
        for proj in self.projections:
            self.projection_dropdown.addItem(proj)
        self.interact_column.addWidget(self.projection_dropdown)

        button_grid = QGridLayout()

        self.extent_button = QPushButton("Extent")
        button_grid.addWidget(self.extent_button, 0, 0)
        self.plot_button = QPushButton("Plot")
        button_grid.addWidget(self.plot_button, 0, 1)
        self.airport_button = QPushButton("Airport")
        button_grid.addWidget(self.airport_button, 1, 0)
        self.reset_button = QPushButton("Reset")
        button_grid.addWidget(self.reset_button, 1, 1)

        self.interact_column.addLayout(button_grid)

        self.area_input_description = QLabel("Area")
        self.area_input = QLineEdit()
        area_input_layout = QHBoxLayout()
        area_input_layout.addWidget(self.area_input_description)
        area_input_layout.addWidget(self.area_input)
        self.interact_column.addLayout(area_input_layout)

        self.area_select = QListWidget()
        self.interact_column.addWidget(self.area_select)

        self.time_slider = QSlider(QtCore.Qt.Horizontal)
        self.time_description = QLabel("Date max.")
        self.time_slider_info = QLabel()
        time_layout = QHBoxLayout()
        time_layout.addWidget(self.time_description)
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_slider_info)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(99)
        self.time_slider.setValue(99)
        self.time_slider.setEnabled(False)
        self.interact_column.addLayout(time_layout)

        self.altitude_slider = QSlider(QtCore.Qt.Horizontal)
        self.altitude_description = QLabel("Altitude max.")
        self.altitude_slider_info = QLabel("60000")
        self.altitude_slider.setSingleStep(5)
        self.altitude_slider.setPageStep(100)
        self.altitude_slider.setMinimum(0)
        self.altitude_slider.setMaximum(600)
        self.altitude_slider.setValue(600)
        altitude_layout = QHBoxLayout()
        altitude_layout.addWidget(self.altitude_description)
        altitude_layout.addWidget(self.altitude_slider)
        altitude_layout.addWidget(self.altitude_slider_info)
        self.interact_column.addLayout(altitude_layout)

        self.identifier_description = QLabel("Callsign/ID")
        self.identifier_input = QLineEdit()

        identifier_layout = QHBoxLayout()
        identifier_layout.addWidget(self.identifier_description)
        identifier_layout.addWidget(self.identifier_input)
        self.interact_column.addLayout(identifier_layout)

        self.identifier_select = QListWidget()
        self.identifier_select.setSelectionMode(3)  # extended selection
        self.interact_column.addWidget(self.identifier_select)

        mainLayout = QGridLayout()
        mainLayout.addLayout(plot_column, 0, 0)
        mainLayout.addLayout(self.interact_column, 0, 1)

        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)

    # -- Callbacks --

    @dont_crash
    def on_time_moved(self, value: int, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        self.time_slider_info.setText(self.date_options[value])

    @dont_crash
    def on_altitude_moved(self, value: int, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        self.altitude_slider_info.setText(f"{100*value}")

    @dont_crash
    def on_select(self, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        if self.traffic is not None:
            max_alt = 100 * self.altitude_slider.value()
            max_time = self.dates[self.time_slider.value()]
            self.on_filter(max_alt, max_time)

    @dont_crash
    def on_filter(self, max_alt: int, max_time: datetime) -> None:

        assert self._traffic is not None

        west, east, south, north = self.map_plot.ax.get_extent(
            crs=PlateCarree()
        )

        self._tview = self._traffic.before(max_time).sort_values("timestamp")

        if self._tview is None:
            return

        filtered = Traffic.from_flights(
            Flight(f.data.ffill().bfill()) for f in self._tview
        )
        if "altitude" in filtered.data.columns:
            filtered = filtered.query(
                f"altitude != altitude or altitude <= {max_alt}"
            )
        if "latitude" in self._tview.data.columns:
            filtered = filtered.query(
                "latitude != latitude or "
                f"({west} <= longitude <= {east} and "
                f"{south} <= latitude <= {north})"
            )

        self.identifier_select.clear()
        text = self.identifier_input.text()
        # cast is necessary because of the @lru_cache on callsigns which hides
        # the type annotation
        for callsign in sorted(cast(Set[str], filtered.callsigns)):
            if re.match(text, callsign, flags=re.IGNORECASE):
                self.identifier_select.addItem(callsign)
        self.map_plot.default_plot(self._tview.subset(filtered.callsigns))
        self.set_float_columns()

    @dont_crash
    def on_extent_button(self, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        if self.area_select.count() == 0:
            if len(self.area_input.text()) == 0:
                self.map_plot.ax.set_global()
            else:
                self.map_plot.ax.set_extent(location(self.area_input.text()))
        else:
            if self.airac_ready:
                self.map_plot.ax.set_extent(
                    airac[self.area_select.item(0).text()]
                )

        self.map_plot.draw()

        if self.traffic is not None:
            max_alt = 100 * self.altitude_slider.value()
            max_time = self.dates[self.time_slider.value()]
            self.on_filter(max_alt, max_time)

    @dont_crash
    def on_id_change(self, *args, **kwargs) -> None:
        assert self._tview is not None
        self.last_interact = datetime.now()

        list_callsigns = list(
            item.text() for item in self.identifier_select.selectedItems()
        )
        selected_y = list(
            item.text() for item in self.y_selector.selectedItems()
        )
        selected_sec_y = list(
            item.text() for item in self.sec_y_selector.selectedItems()
        )
        self.map_plot.plot_callsigns(self._tview, list_callsigns)
        self.time_plot.create_plot()
        self.time_plot.plot_callsigns(
            self._tview,
            list_callsigns,
            y=selected_y + selected_sec_y,
            secondary_y=selected_sec_y,
        )

    @dont_crash
    def on_id_input(self, text, *args, **kwargs) -> None:
        assert self._tview is not None
        self.last_interact = datetime.now()
        # segfault prone when interactive (decoder)
        # selected = list(
        #     item.text() for item in self.identifier_select.selectedItems()
        # )
        self.identifier_select.clear()
        for callsign in sorted(cast(Set[str], self._tview.callsigns)):
            if re.match(text, callsign, flags=re.IGNORECASE):
                self.identifier_select.addItem(callsign)
                # if callsign in selected:
                #     curItem = self.identifier_select.item(
                #         self.identifier_select.count() - 1
                #     )
                #     self.identifier_select.setItemSelected(curItem, True)

    @dont_crash
    def on_plot_button(self, *args, **kwargs) -> None:
        assert self._tview is not None
        self.last_interact = datetime.now()
        if self.area_select.count() == 0:
            if len(self.area_input.text()) == 0:
                self.map_plot.default_plot(self._tview)
            else:
                location(self.area_input.text()).plot(
                    self.map_plot.ax, color="grey", linestyle="dashed"
                )
        else:
            if self.airac_ready:
                selected = self.area_select.selectedItems()
                if len(selected) == 0:
                    return
                airspace = airac[selected[0].text()]
                if airspace is not None:
                    airspace.plot(self.map_plot.ax)

        self.map_plot.draw()

    @dont_crash
    def on_area_input(self, text: str, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        self.area_select.clear()
        if len(text) > 0 and self.airac_ready:
            for airspace_info in airac.parse(text):
                self.area_select.addItem(airspace_info.name)

    @dont_crash
    def on_area_select(self, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        selected = self.area_select.selectedItems()
        if len(selected) == 0:
            return
        if self.airac_ready:
            airspace = airac[selected[0].text()]
            if airspace is not None:
                self.map_plot.ax.set_extent(airspace)
                self.map_plot.draw()

    @dont_crash
    def on_plot_airport(self, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        if len(self.area_input.text()) == 0:
            from cartotools.osm import request, tags

            west, east, south, north = self.map_plot.ax.get_extent(
                crs=PlateCarree()
            )
            if abs(east - west) > 1 or abs(north - south) > 1:
                # that would be a too big request
                return
            request((west, south, east, north), **tags.airport).plot(
                self.map_plot.ax
            )
        else:
            from traffic.data import airports

            airport = airports[self.area_input.text()]
            if airport is not None:
                airport.plot(self.map_plot.ax)
        self.map_plot.draw()

    @dont_crash
    def on_clear_button(self, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        if self.traffic is not None:
            assert self._traffic is not None
            self._tview = self._traffic.sort_values("timestamp")
            self.altitude_slider.setValue(600)
            self.make_map(self.projection_dropdown.currentIndex())
            self.time_plot.create_plot()
            self.set_float_columns()

    @dont_crash
    def make_map(self, index_projection: int, *args, **kwargs) -> None:
        self.last_interact = datetime.now()
        self.map_plot.create_map(self.projections[index_projection])
        if self._tview is not None:
            self.map_plot.default_plot(self._tview)

    @dont_crash
    def on_open(self, index: int, *args, **kwargs) -> None:
        if self.decoder is not None and self.updateTraffic is not None:
            self.updateTraffic.terminate()
            self.decoder.stop()
        if index == 0:
            self.openFile()
        elif index == 1:
            self.openDump1090()
        else:
            address = config.get("decoders", self.open_options[index])
            host_port, reference = address.split("/")
            host, port = host_port.split(":")
            self.decoder = ModeS_Decoder.from_address(
                host, int(port), reference
            )
            refresh_time = config.getint(
                "decoders", "refresh_time", fallback=30
            )
            self.updateTraffic = UpdateTraffic(self, refresh_time)
            self.updateTraffic.start()

    # -- Basic setters --

    def set_icons(self) -> None:

        logging.info("Setting options")
        icon_path = Path(__file__).absolute().parent.parent.parent / "icons"

        if sys.platform == "linux":
            icon_full = QtGui.QIcon((icon_path / "travel-white.svg").as_posix())
        else:
            icon_full = QtGui.QIcon((icon_path / "travel-grey.svg").as_posix())

        self.setWindowIcon(icon_full)

    def set_time_range(self) -> None:
        assert self._traffic is not None
        self.time_slider.setEnabled(True)
        self.dates = [
            self._traffic.start_time
            + i * (self._traffic.end_time - self._traffic.start_time) / 99
            for i in range(100)
        ]

        tz_now = datetime.now().astimezone().tzinfo
        if self._traffic.start_time.tzinfo is not None:
            self.date_options = [
                t.tz_convert("utc").strftime("%H:%M") for t in self.dates
            ]
        else:
            self.date_options = [
                t.tz_localize(tz_now).tz_convert("utc").strftime("%H:%M")
                for t in self.dates
            ]
        self.time_slider_info.setText(self.date_options[-1])

    def set_float_columns(self) -> None:
        assert self._traffic is not None
        self.y_selector.clear()
        self.sec_y_selector.clear()
        for column, dtype in self._traffic.data.dtypes.items():
            if column not in ("latitude", "longitude"):
                if dtype in ["float64", "int64"]:
                    self.y_selector.addItem(column)
                    self.sec_y_selector.addItem(column)

    def openDump1090(self) -> None:
        reference, ok = QInputDialog.getText(
            self, "dump1090 reference", "Reference airport:"
        )

        if ok:
            self.open_dropdown.setItemText(1, f"dump1090 ({reference})")
            self.decoder = ModeS_Decoder.from_dump1090(reference)
            refresh_time = config.getint(
                "decoders", "refresh_time", fallback=30
            )
            self.updateTraffic = UpdateTraffic(self, refresh_time)
            self.updateTraffic.start()

    @dont_crash
    def openFile(self, *args, **kwargs) -> None:
        options = {
            "caption": "Open file",
            "filter": (
                "Pandas DataFrame (*.pkl);;"
                "CSV files (*.csv);;"
                "Sqlite3 files (*.db)"
            ),
            #             "filter": "Data files (*.csv *.pkl)",
            "directory": os.path.expanduser("~"),
        }

        self.filename = QFileDialog.getOpenFileName(self, **options)[0]
        if self.filename == "":
            return
        self.filename = Path(self.filename)
        self._traffic = Traffic.from_file(self.filename)

        assert self._traffic is not None
        self._tview = self._traffic.sort_values("timestamp")
        assert self._tview is not None
        self.open_dropdown.setItemText(0, self.filename.name)
        self.map_plot.default_plot(self._tview)

        self.identifier_select.clear()
        for callsign in sorted(cast(Set[str], self._tview.callsigns)):
            self.identifier_select.addItem(callsign)

        self.set_time_range()
        self.set_float_columns()


def main():

    if sys.platform == "win32":
        # This lets you keep your custom icon in the Windows taskbar
        import ctypes

        myappid = "org.xoolive.traffic"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    main = MainScreen()
    main.show()

    return app.exec_()
