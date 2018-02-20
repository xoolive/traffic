from datetime import timedelta

from tqdm import tqdm
from traffic import kml
from traffic.data import sectors
from traffic.so6 import SO6

so6 = SO6.parse_pkl("20160101_20160101_0000_2359_____m3.pkl")

# This will help to select callsigns (only one line per flight because of `at`)
bdx_noon_flights = (so6.at("2016/01/01 12:00").
                    inside_bbox(sectors['LFBBBDX']).
                    intersects(sectors['LFBBBDX']))

# Trajectoire des vols entre midi et 12h30, pour ceux qui sont à midi dans le
# secteur
so6_interval = so6.between("2016/01/01 12:00", timedelta(minutes=30))
so6_bdx = so6_interval.select(bdx_noon_flights.callsigns)


with kml.export('export.kml') as fh:
    fh.append(sectors['LFBBBDX'].export_kml(color='blue', alpha=.3))
    # iterate on so6 yield callsign, flight
    for callsign, flight in tqdm(so6_bdx):
        fh.append(flight.export_kml(color='#aa3a3a'))
