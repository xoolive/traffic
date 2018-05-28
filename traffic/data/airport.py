import pickle
import re
from pathlib import Path
from typing import NamedTuple, Optional


class __airport__nt(NamedTuple):

    alt: int
    country: str
    iata: str
    icao: str
    lat: float
    lon: float
    name: str

class Airport(__airport__nt):

    def __repr__(self):
        return (
            f"{self.icao}/{self.iata}    {self.name.strip()} ({self.country})"
            f"\n\t{self.lat} {self.lon} altitude: {self.alt}")


class AirportParser(object):

    cache: Optional[Path] = None

    def __init__(self):
        if self.cache is not None and self.cache.exists():
            with open(self.cache, 'rb') as fh:
                self.airports = pickle.load(fh)
        else:
            from .flightradar24 import FlightRadar24  # typing: ignore
            self.airports = FlightRadar24().get_airports()
            if self.cache is not None:
                with open(self.cache, 'wb') as fh:
                    pickle.dump(self.airports, fh)

    def __getitem__(self, name: str):
        return next((a for a in self.airports
                     if (a.iata == name.upper()) or (a.icao == name.upper())),
                     None)

    def search(self, name: str):
        return list((a for a in self.airports
                     if (a.iata == name.upper()) or (a.icao == name.upper())
                     or (re.match(name, a.country, flags=re.IGNORECASE))
                     or (re.match(name, a.name, flags=re.IGNORECASE))))
