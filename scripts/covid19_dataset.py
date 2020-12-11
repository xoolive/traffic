import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd

query = """
select sv.icao24, sv.callsign,
sv.lat as lat2, sv.lon as lon2, sv.baroaltitude as alt2, sv.hour,
est.firstseen, est.estdepartureairport, est.lastseen, est.estarrivalairport,
est.day, est.lat1, est.lon1, est.alt1
from state_vectors_data4 as sv join (
    select icao24 as e_icao24, firstseen, lastseen,
    estdepartureairport, estarrivalairport, callsign as e_callsign, day,
    t.time as t1, t.latitude as lat1, t.longitude as lon1, t.altitude as alt1
    from flights_data4, flights_data4.track as t
    where ({before_hour} <= day and day <= {after_hour}) and t.time = firstseen
) as est
on sv.icao24 = est.e_icao24 and sv.callsign = est.e_callsign and
est.lastseen = sv.time
where hour>={before_hour} and hour<{after_hour} and
time>={before_time} and time<{after_time}
"""


@click.command()
@click.argument("start", type=str)
@click.argument("stop", type=str)
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-c",
    "--cached",
    is_flag=True,
    default=False,
    show_default=True,
    help="Reuse already downloaded data",
)
@click.option(
    "-f",
    "--flight_db",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
)
@click.option("-v", "--verbose", count=True, help="Verbosity level")
def main(
    start: str,
    stop: str,
    output_dir: Path,
    *args,
    flight_db: Optional[Path] = None,
    cached: bool = False,
    verbose: int = 0,
):
    click.echo("Downloading covid19 dataset data...")

    logger = logging.getLogger()
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose > 1:
        logger.setLevel(logging.DEBUG)

    from traffic.data import aircraft, opensky

    df = opensky.request(
        query,
        start,
        stop,
        date_delta=timedelta(days=1),
        columns=[
            "icao24",
            "callsign",
            "latitude_2",
            "longitude_2",
            "altitude_2",
            "hour",
            "firstseen",
            "estdepartureairport",
            "lastseen",
            "estarrivalairport",
            "day",
            "latitude_1",
            "longitude_1",
            "altitude_1",
        ],
        cached=cached,
    )

    df = (
        df.sort_values("lastseen")
        .drop(columns=["hour"])
        .rename(
            columns=dict(
                estarrivalairport="destination", estdepartureairport="origin",
            )
        )
        .assign(
            firstseen=lambda df: pd.to_datetime(
                df.firstseen, unit="s",
            ).dt.tz_localize("utc"),
            lastseen=lambda df: pd.to_datetime(
                df.lastseen, unit="s",
            ).dt.tz_localize("utc"),
            day=lambda df: pd.to_datetime(df.day, unit="s",).dt.tz_localize(
                "utc"
            ),
        )
    )
    df_merged = df.merge(aircraft.opensky_db, how="left")[
        [
            "callsign",
            "icao24",
            "registration",
            "typecode",
            "origin",
            "destination",
            "firstseen",
            "lastseen",
            "day",
            "latitude_1",
            "longitude_1",
            "altitude_1",
            "latitude_2",
            "longitude_2",
            "altitude_2",
        ]
    ].query("callsign == callsign and firstseen == firstseen")

    if flight_db is not None:
        # Flight database from Jannis
        flightdb = (
            pd.read_csv(flight_db)
            .rename(columns=dict(Callsign="callsign"))
            .assign(
                valid=lambda df: pd.to_datetime(
                    df.ValidFrom, unit="s"
                ).dt.tz_localize("utc"),
                number=lambda df: df.OperatorIata + df.FlightNumber.astype(str),
            )[["callsign", "Route", "valid", "number"]]
        )

        df_full = df_merged.merge(flightdb, how="left")
        df = pd.concat(
            [
                df_full.query("firstseen < valid or valid != valid").drop(
                    columns=["Route", "valid", "number"]
                ),
                df_full.query("firstseen >= valid").drop(
                    columns=["Route", "valid"]
                ),
            ]
        )
    else: 
        df = df_merged.assign(number=None)

    df_clean = df.sort_values("firstseen")[
        [
            "callsign",
            "number",
            "icao24",
            "registration",
            "typecode",
            "origin",
            "destination",
            "firstseen",
            "lastseen",
            "day",
            "latitude_1",
            "longitude_1",
            "altitude_1",
            "latitude_2",
            "longitude_2",
            "altitude_2",
        ]
    ]

    start = df_clean.lastseen.min()
    stop = df_clean.firstseen.max()

    filename: Path = Path(output_dir)
    filename = filename / f"flightlist_{start:%Y%m%d}_{stop:%Y%m%d}.csv.gz"
    df_clean.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
