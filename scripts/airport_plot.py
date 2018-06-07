import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

import maya
from tqdm import tqdm
from traffic.core import Traffic
from traffic.core.time import timelike
from traffic.data import airports, opensky
from traffic.data.basic.airport import Airport
from traffic.drawing import UTM, countries, lakes, rivers


def get_data(before_time: timelike, after_time: timelike,
             airport: Airport) -> Optional[Traffic]:

    logging.info(f"Getting callsigns arriving at {airport.name}")
    callsigns = opensky.at_airport(before_time, after_time, airport)
    if callsigns is None or callsigns.shape[0] == 0:
        return None

    callsigns_list = callsigns.loc[callsigns.callsign.notnull(), 'callsign']
    logging.info(f"{len(callsigns_list)} callsigns found")
    logging.info(f"Fetching data from OpenSky database")

    flights = opensky.history(before_time, after_time,
                              callsign=callsigns_list,
                              progressbar=tqdm)

    return flights


def plot_data(flights: Traffic, airport: Airport, output: Path,
              arrival: int=1):

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=UTM((airport.lon + 180) // 6,
                                 southern_hemisphere=(airport.lat > 0)))

    ax.add_feature(countries(scale='50m'))
    ax.add_feature(rivers(scale='50m'))
    ax.add_feature(lakes(scale='50m'))
    ax.gridlines()
    ax.set_extent(airport.extent)

    try:
        from cartotools.osm import request, tags
        request(f'{airport.icao} airport', **tags.airport).plot(ax)
    except Exception:
        pass

    fd = flights.data

    fd = fd[fd.longitude > airport.lon - 1]
    fd = fd[fd.longitude < airport.lon + 1]
    fd = fd[fd.latitude > airport.lat - 1]
    fd = fd[fd.latitude < airport.lat + 1]
    fd = fd.groupby('callsign').filter(
        lambda f: arrival * f.vertical_rate[:-100].mean() < -10)

    for flight in Traffic(fd):
        flight.plot(ax)

    fig.suptitle(f"{airport.name}")
    fig.set_tight_layout(True)

    logging.info(f"Image saved as {output}")
    fig.savefig(output.as_posix())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce a trajectory plot at a given airport")

    parser.add_argument("airport",
                        help="ICAO or IATA code of an airport")

    parser.add_argument("time", default="now",
                        help="day/time of screenshot")

    parser.add_argument("-o", dest="output", type=Path,
                        help="output file")

    parser.add_argument("-d", dest="departure", action="store_true",
                        help="show departure instead of arrival map")

    parser.add_argument("-v", dest="verbose", action="count", default=0,
                        help="display logging messages")

    args = parser.parse_args()
    airport = airports[args.airport]
    if args.output is None:
        raise RuntimeError("Please specify an output file with option -o")

    if args.time == 'now':
        after_time = maya.parse("now") - timedelta(hours=2)  # type: ignore
    else:
        after_time = maya.parse(args.time)

    if args.verbose == 1:
        logging.basicConfig(format="%(levelname)s: %(message)s",
                            level=logging.INFO)
    elif args.verbose >= 2:
        logging.basicConfig(format="%(levelname)s: %(message)s",
                            level=logging.DEBUG)

    before_time = after_time - timedelta(minutes=30)

    flights = get_data(before_time, after_time, airport)
    if flights is not None:
        plot_data(flights, airport, args.output, -1 if args.departure else 1)
    else:
        logging.info("No data for this timeframe")
