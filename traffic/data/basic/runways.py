import pickle
from io import BytesIO
from pathlib import Path
from typing import Dict, NamedTuple, Optional
from zipfile import ZipFile

import requests


class Threshold(NamedTuple):
    lat: float
    lon: float
    name: str

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, name):
        if name == "latitude":
            return self.lat
        if name == "longitude":
            return self.lon
        if name == "altitude":
            return self.alt


__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "ProfHoekstra/bluesky/master/data/navdata"


class Runways(object):

    cache: Optional[Path] = None

    def __init__(self) -> None:
        if self.cache is not None and self.cache.exists():
            with self.cache.open("rb") as fh:
                self.runways = pickle.load(fh)
        else:
            self.initialize()
            if self.cache is not None:
                with self.cache.open("wb") as fh:
                    pickle.dump(self.runways, fh)

    def initialize(self):
        self.runways = dict()
        curthresholds: Dict[str, Threshold] = dict()

        c = requests.get(base_url + "/apt.zip")

        with ZipFile(BytesIO(c.content)).open("apt.dat", "r") as fh:
            for line in fh.readlines():
                elems = (
                    line.decode(encoding="ascii", errors="ignore")
                    .strip()
                    .split()
                )
                if len(elems) == 0:
                    continue

                # 1: AIRPORT
                if elems[0] == "1":
                    # Add airport to runway threshold database
                    curthresholds = dict()
                    self.runways[elems[4]] = curthresholds

                if elems[0] == "100":
                    # Only asphalt and concrete runways
                    if int(elems[2]) > 2:
                        continue

                    lat0 = float(elems[9])
                    lon0 = float(elems[10])
                    # offset0 = float(elems[11])

                    lat1 = float(elems[18])
                    lon1 = float(elems[19])
                    # offset1 = float(elems[20])

                    # threshold information:
                    #       ICAO code airport,
                    #       Runway identifier,
                    #       latitude, longitude, bearing
                    # vertices: gives vertices of the box around the threshold

                    # opposite runways are on the same line.
                    #       RWY1: 8-11, RWY2: 17-20
                    # Hence, there are two thresholds per line
                    # thr0: First lat0 and lon0, then lat1 and lat1, offset=[11]
                    # thr1: First lat1 and lat1, then lat0 and lon0, offset=[20]

                    thr0 = Threshold(lat0, lon0, elems[8])
                    thr1 = Threshold(lat1, lon1, elems[17])
                    curthresholds[elems[8]] = thr0
                    curthresholds[elems[17]] = thr1
