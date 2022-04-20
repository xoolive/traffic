from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
from flask import Flask

import pandas as pd
from traffic import config
from traffic.data import ModeS_Decoder
from traffic.data.adsb.decode import Entry

if TYPE_CHECKING:
    from traffic.core.structure import Airport


class Decode(ModeS_Decoder):
    def __init__(
        self,
        reference: None | str | Airport | tuple[float, float] = None,
    ) -> None:
        super().__init__(
            reference,
            expire_frequency=pd.Timedelta("10 seconds"),
            expire_threshold=pd.Timedelta("10 seconds"),
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
        print("do_something")


@click.command()
@click.argument("source")
@click.option("--reference", "initial_reference", default="LFBO")
@click.option("--filename", default="~/ADSB_EHS_RAW_%Y%m%d.csv")
@click.option("--host", "serve_host", default="127.0.0.1")
@click.option("--port", "serve_port", default=5050, type=int)
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
        decoder = Decode.from_dump1090(initial_reference, dump_file)
    else:
        address = config.get("decoders", source)
        host_port, reference = address.split("/")
        host, port = host_port.split(":")
        decoder = Decode.from_address(host, int(port), reference, dump_file)

    app = Flask(__name__)

    @app.route("/")
    def home() -> dict[str, int]:
        d = dict(decoder.acs)
        return dict((key, len(aircraft.cumul)) for (key, aircraft) in d.items())

    @app.route("/icao24/<icao24>")
    def get_icao24(icao24: str) -> dict[str, list[Entry]]:
        d = dict(decoder.acs)
        aircraft_or_none = d.get(icao24, None)
        if aircraft_or_none:
            return {
                icao24: list(
                    entry
                    | dict(
                        timestamp=entry["timestamp"].timestamp()  # type: ignore
                    )
                    for entry in aircraft_or_none.cumul
                )
            }
        else:
            return {icao24: []}

    app.run(serve_host, serve_port)


if __name__ == "__main__":
    main()
