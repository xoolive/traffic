import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

import maya
import pandas as pd
from cartopy.crs import UTM, EuroPP, PlateCarree
from cartopy.feature import NaturalEarthFeature
from traffic.core.time import timelike
from traffic.data import airports, opensky
from traffic.data.airport import Airport


def get_data(before_time: timelike, after_time: timelike,
             airport: Airport) -> Optional[pd.DataFrame]:

    logging.info(f"Getting callsigns arriving at {airport.name}")
    callsigns = opensky.at_airport(before_time, after_time, airport)
    if callsigns is None or callsigns.shape[0] == 0:
        return None

    callsigns_list = callsigns.loc[callsigns.callsign.notnull(), 'callsign']
    logging.info(f"{len(callsigns_list)} callsigns found")
    logging.info(f"Fetching data from OpenSky database")

    flights = opensky.history(before_time, after_time, callsign=callsigns_list)
    flights = flights[~flights.onground] # oh!
    flights = flights.sort_values('timestamp')

    return flights


def plot_data(flights: pd.DataFrame, airport: Airport, output: Path,
              arrival: int=1):

    countries = NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
        edgecolor='#524c50',
        facecolor='none',
        alpha=.5)


    rivers = NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale='10m',
        edgecolor='#226666',
        facecolor='none',
        alpha=.5
    )

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=UTM((airport.lon + 180) // 6,
                                 southern_hemisphere=(airport.lat > 0)))

    ax.add_feature(countries)
    ax.add_feature(rivers)
    ax.gridlines()
    ax.set_extent((airport.lon - 1, airport.lon + 1,
                   airport.lat - .7, airport.lat + .7))

    try:
        from cartotools.osm import location, request, tags
        from shapely.geometry import base

        logging.info(f"Getting city background from OpenStreetMap")
        ax.add_geometries([location[airport.name.split()[0]].shape],
                          crs=PlateCarree(), linestyle='dashed',
                          facecolor='none', edgecolor="gray",)

        logging.info(f"Getting airport background from OpenStreetMap")
        r = request.json_request(location[airport.icao], **tags.airport)
        geo = r.geometry()
        if geo.area < 1e-4:
            r = request.json_request(location[airport.name], **tags.airport)
            geo = r.geometry()
        if not isinstance(geo, base.BaseMultipartGeometry):
            geo = [geo]
        ax.add_geometries(geo, crs=PlateCarree(),
                          edgecolor="#aa3aaa", facecolor='None', )
    except:
        pass

    flights = flights[flights.longitude > airport.lon - 1]
    flights = flights[flights.longitude < airport.lon + 1]
    flights = flights[flights.latitude > airport.lat - 1]
    flights = flights[flights.latitude < airport.lat + 1]
    flights = flights.groupby('callsign').filter(
        lambda f: arrival * f.vertical_rate[:-100].mean() < -10)

    for cs, flight in flights.groupby('callsign'):
        ax.plot(flight.longitude, flight.latitude,
                transform=PlateCarree(), color='#3aaa3a', lw=.7)

    fig.suptitle(f"{airport.name}")
    fig.set_tight_layout(True)

    fig.savefig(output.as_posix())

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce a chevelu plot at a given airport")

    parser.add_argument("airport",
                        help="ICAO or IATA code of an airport")

    parser.add_argument("time", default="now",
                        help="day/time of screenshot")

    parser.add_argument("-o", dest="output", type=Path,
                        help="output file")

    parser.add_argument("-d", dest="departure", action="store_true",
                        help="show departure instead of arrival map")

    parser.add_argument("-v", dest="verbose", action="count",
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
