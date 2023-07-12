# mypy: ignore-errors
# TODO The line above avoids a typing error only visible in Github Actions

from __future__ import annotations

import base64
import logging
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import click
from flask import Flask
from flask_cors import CORS
from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.table import Table
from textual.app import App, ComposeResult
from textual.widget import Widget
from waitress import serve

import pandas as pd
from traffic import config
from traffic.data import ModeS_Decoder, aircraft
from traffic.data.adsb.decode import Entry

if TYPE_CHECKING:
    from traffic.core.structure import Airport


class Decoder(ModeS_Decoder):
    instance: Decoder

    def __init__(
        self,
        reference: None | str | Airport | tuple[float, float] = None,
    ) -> None:
        super().__init__(
            reference,
            expire_frequency=pd.Timedelta("10 seconds"),
            expire_threshold=pd.Timedelta("10 seconds"),
        )

        Decoder.instance = self

    def on_expire_aircraft(self, icao: str) -> None:
        logging.info(f"expire aircraft {icao}")
        return super().on_expire_aircraft(icao)

    def on_new_aircraft(self, icao24: str) -> None:
        logging.info(f"new aircraft {icao24}")

    # @ModeS_Decoder.on_timer("5s")
    # def do_something(self) -> None:
    #     logging.info("do_something")


class AircraftListWidget(Widget):
    decoder: None | ModeS_Decoder = None

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh)

    def render(self) -> Table:
        table = Table(show_lines=False, box=SIMPLE_HEAVY)
        for column in [
            "icao24",
            "aircraft",
            "callsign",
            "latitude",
            "longitude",
            "altitude",
            "speed",
            "track",
            "count",
            "last_seen",
        ]:
            table.add_column(column)

        if self.decoder is None:
            return table

        acs = sorted(
            (ac for icao, ac in self.decoder.acs.items()),
            key=lambda aircraft: len(aircraft.cumul),
            reverse=True,
        )
        for ac_elt in acs:
            cumul = list(ac_elt.cumul)
            if len(cumul) > 1:
                tail = aircraft.get_unique(ac_elt.icao24)
                table.add_row(
                    ac_elt.icao24,
                    format(tail, "%typecode %registration") if tail else "",
                    ac_elt.callsign,
                    str(ac_elt.lat),
                    str(ac_elt.lon),
                    str(ac_elt.alt),
                    str(ac_elt.spd),
                    str(ac_elt.trk),
                    str(len(cumul)),
                    format(cumul[-1]["timestamp"], "%H:%M:%S"),
                )

        return table


class SimpleApp(App):
    aircraft_widget: AircraftListWidget = AircraftListWidget()
    flask_thread: threading.Thread | None = None

    BINDINGS = [  # noqa: RUF012
        ("escape,q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield self.aircraft_widget


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def home() -> dict[str, int]:
    decoder = Decoder.instance
    d = dict(decoder.acs)
    return dict((key, len(aircraft.cumul)) for (key, aircraft) in d.items())


@app.route("/traffic")
def get_all() -> dict[str, str]:
    decoder = Decoder.instance
    t = decoder.traffic
    pickled_traffic = base64.b64encode(pickle.dumps(t)).decode("utf-8")
    return {"traffic": pickled_traffic}


@app.route("/icao24/<icao24>")
def get_icao24(icao24: str) -> dict[str, list[Entry]]:
    decoder = Decoder.instance
    d = dict(decoder.acs)
    aircraft_or_none = d.get(icao24, None)
    if aircraft_or_none:
        return {
            icao24: list(
                entry
                | dict(timestamp=entry["timestamp"].timestamp())  # type: ignore
                for entry in aircraft_or_none.cumul
            )
        }
    else:
        return {icao24: []}


@click.command()
@click.argument("source")
@click.option(
    "-r",
    "--reference",
    "initial_reference",
    help="Reference position (airport code)",
)
@click.option(
    "-f",
    "--filename",
    default="~/ADSB_EHS_RAW_%Y%m%d.csv",
    show_default=True,
    help="Filename pattern describing where to dump raw data",
)
@click.option(
    "--host",
    "serve_host",
    show_default=True,
    default="127.0.0.1",
    help="host address where to serve decoded information",
)
@click.option(
    "--port",
    "serve_port",
    show_default=True,
    default=5050,
    type=int,
    help="port to serve decoded information",
)
@click.option(
    "-l",
    "--log",
    "log_file",
    default=None,
    help="logging information",
)
@click.option(
    "--tui",
    is_flag=True,
    show_default=True,
    default=False,
    help="Display aircraft table in text user interface mode",
)
@click.option("-v", "--verbose", count=True, help="Verbosity level")
def main(
    source: str,
    initial_reference: str | None = None,
    filename: str | Path = "~/ADSB_EHS_RAW_%Y%m%d_tcp.csv",
    decode_uncertainty: bool = False,
    expire_aircraft: int | None = None,
    update_reference: int | None = None,
    serve_host: str | None = "127.0.0.1",
    serve_port: int | None = 5050,
    log_file: str | None = None,
    tui: bool = True,
    verbose: int = 0,
) -> None:
    """Decode ADS-B data from a source of raw binary data.

    The source parameter may be one of "dump1090", "rtlsdr", one entry in the
    decoders section of the configuration file or a file name where data is
    dumped.

    """

    logger = logging.getLogger()
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose > 1:
        logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    dump_file = Path(filename).with_suffix(".csv").as_posix()

    if Path(source).expanduser().exists():
        assert initial_reference is not None
        decoder = Decoder.from_file(source, initial_reference)
        traffic = decoder.traffic
        if traffic is not None:
            console = Console()
            console.print(traffic)
            icao24 = traffic.basic_stats.reset_index().iloc[0].icao24
            flight = traffic[icao24]
            console.print(flight)
            output = Path(source).expanduser().with_suffix(".parquet")
            traffic.to_parquet(output)
            logging.warning(f"File written: {output}")
        return None

    if source == "dump1090":
        assert initial_reference is not None
        decoder = Decoder.from_dump1090(
            initial_reference, dump_file, uncertainty=decode_uncertainty
        )
    elif source == "rtlsdr":
        assert initial_reference is not None
        decoder = Decoder.from_rtlsdr(
            initial_reference, dump_file, uncertainty=decode_uncertainty
        )
    elif config.has_section(f"decoders.{source}"):
        host = config.get(f"decoders.{source}", "host")
        port = config.getint(f"decoders.{source}", "port")
        reference = config.get(f"decoders.{source}", "reference")
        socket_option = (
            config.get(f"decoders.{source}", "socket")
            if config.has_option(f"decoders.{source}", "socket")
            else "TCP"
        )
        time_fmt = (
            config.get(f"decoders.{source}", "time_fmt")
            if config.has_option(f"decoders.{source}", "time_fmt")
            else "default"
        )
        file_pattern = (
            config.get(f"decoders.{source}", "file")
            if config.has_option(f"decoders.{source}", "file")
            else dump_file
        )
        decoder = Decoder.from_address(
            host=host,
            port=port,
            reference=reference,
            file_pattern=file_pattern,
            uncertainty=decode_uncertainty,
            tcp=socket_option.upper() == "TCP",
            time_fmt=time_fmt,
        )
    else:
        raise RuntimeError(f"No decoders.{source} section found")

    flask_thread = threading.Thread(
        target=serve,
        daemon=True,
        kwargs=dict(
            app=app,
            host=serve_host,
            port=serve_port,
            threads=8,
        ),
    )
    flask_thread.start()

    if tui:
        tui_app = SimpleApp()
        tui_app.aircraft_widget.decoder = decoder
        tui_app.flask_thread = flask_thread
        tui_app.run()
    else:
        flask_thread.join()


if __name__ == "__main__":
    main()
