from __future__ import annotations
import base64

import logging
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from flask import Flask, current_app
from rich.box import SIMPLE_HEAVY
from rich.table import Table
from textual.app import App
from textual.widget import Widget

import pandas as pd
from traffic import config
from traffic.data import ModeS_Decoder, aircraft

if TYPE_CHECKING:
    from traffic.core.structure import Airport


class Decode(ModeS_Decoder):
    def __init__(
        self,
        reference: None | str | Airport | tuple[float, float] = None,
    ) -> None:
        super().__init__(
            reference,
            expire_frequency=pd.Timedelta("1 minute"),
            expire_threshold=pd.Timedelta("10 minutes"),
        )
        # uncertainty
        # dump file

    def on_expire_aircraft(self, icao: str) -> None:
        logging.info(f"expire aircraft {icao}")
        return super().on_expire_aircraft(icao)

    def on_new_aircraft(self, icao24: str) -> None:
        logging.info(f"new aircraft {icao24}")

    @ModeS_Decoder.on_timer("5s")
    def do_something(self) -> None:
        logging.info("do_something")


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
        for a in acs:
            cumul = list(a.cumul)
            if len(cumul) > 1:
                tail = aircraft.get_unique(a.icao24)
                table.add_row(
                    a.icao24,
                    format(tail, "%typecode %registration") if tail else "",
                    a.callsign,
                    str(a.lat),
                    str(a.lon),
                    str(a.alt),
                    str(a.spd),
                    str(a.trk),
                    str(len(cumul)),
                    format(cumul[-1]["timestamp"], "%H:%M:%S"),
                )

        return table


class SimpleApp(App):
    aircraft_widget: AircraftListWidget = AircraftListWidget()
    flask_thread: threading.Thread | None = None

    async def on_load(self, event: Any) -> None:
        await self.bind("q", "quit")

    async def on_mount(self) -> None:
        await self.view.dock(self.aircraft_widget)


app = Flask(__name__)


@app.route("/")
def home() -> dict[str, int]:
    d = dict(current_app.decoder.acs)
    return dict((key, len(aircraft.cumul)) for (key, aircraft) in d.items())


@app.route("/traffic")
def get_all() -> dict[str, str]:
    t = current_app.decoder.traffic
    pickled_traffic = base64.b64encode(pickle.dumps(t)).decode('utf-8')
    return {"traffic": pickled_traffic}


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
    default="0.0.0.0",
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
    tui: bool = True,
    verbose: int = 0,
) -> None:

    logger = logging.getLogger()
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose > 1:
        logger.setLevel(logging.DEBUG)

    dump_file = Path(filename).with_suffix(".csv").as_posix()

    if source == "dump1090":
        assert initial_reference is not None
        app.decoder = Decode.from_dump1090(
            initial_reference, dump_file, uncertainty=decode_uncertainty
        )
    elif source == "rtlsdr":
        assert initial_reference is not None
        app.decoder = Decode.from_rtlsdr(
            initial_reference, dump_file, uncertainty=decode_uncertainty
        )
    else:
        address = config.get("decoders", source)
        host_port, reference = address.split("/")
        host, port = host_port.split(":")
        app.decoder = Decode.from_address(
            host=host,
            port=int(port),
            reference=reference,
            file_pattern=dump_file,
            uncertainty=decode_uncertainty,
        )
    flask_thread = threading.Thread(
        target=app.run,
        kwargs=dict(
            host=serve_host,
            port=serve_port,
            threaded=True,
            debug=False,
            use_reloader=False,
        ),
    )
    flask_thread.start()

    if tui:
        tui_app = SimpleApp()
        tui_app.aircraft_widget.decoder = app.decoder
        tui_app.flask_thread = flask_thread
        tui_app.run()
    else:
        flask_thread.join()


if __name__ == "__main__":
    main()
