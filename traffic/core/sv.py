from typing import TYPE_CHECKING, Iterator, Set

import pandas as pd

from .flight import Position
from .mixins import GeographyMixin

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxesSubplot
    from matplotlib.artist import Artist


class StateVectors(GeographyMixin):
    """Plots the state vectors returned by OpenSky REST API."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def _ipython_key_completions_(self) -> Set[str]:
        return {*self.aircraft, *self.callsigns}

    @property
    def aircraft(self):
        return set(self.data.icao24)

    @property
    def callsigns(self):
        return set(self.data.callsign)

    def __iter__(self) -> Iterator[Position]:
        for _, p in self.data.assign(
            # let's keep this name for now
            name=self.data.callsign
        ).iterrows():
            # TODO work on a clean __repr__ for the Position
            yield Position(p)

    def plot(
        self, ax: "GeoAxesSubplot", s: int = 10, **kwargs
    ) -> "Artist":  # coverage: ignore
        """Plotting function. All arguments are passed to ax.scatter"""
        from cartopy.crs import PlateCarree

        return ax.scatter(
            self.data.longitude,
            self.data.latitude,
            s=s,
            transform=PlateCarree(),
            **kwargs,
        )
