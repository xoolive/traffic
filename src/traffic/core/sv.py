from typing import TYPE_CHECKING, Any, Iterator, Set

from ipyleaflet import MarkerCluster as LeafletMarkerCluster

import pandas as pd

from .flight import Position
from .mixins import GeographyMixin

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.artist import Artist


class StateVectors(GeographyMixin):
    """Plots the state vectors returned by OpenSky REST API."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def _ipython_key_completions_(self) -> Set[str]:
        return {*self.aircraft, *self.callsigns}

    @property
    def aircraft(self) -> Set[str]:
        return set(self.data.icao24)

    @property
    def callsigns(self) -> Set[str]:
        return set(self.data.callsign)

    def __iter__(self) -> Iterator[Position]:
        for _, p in self.data.assign(
            # let's keep this name for now
            name=self.data.callsign
        ).iterrows():
            # TODO work on a clean __repr__ for the Position
            yield Position(p)

    def leaflet(sv: "StateVectors", **kwargs: Any) -> LeafletMarkerCluster:
        """Returns a Leaflet layer to be directly added to a Map.

        The elements passed as kwargs as passed as is to the Marker constructor.
        """
        point_list = list(p.leaflet(title=p.callsign, **kwargs) for p in sv)
        return LeafletMarkerCluster(markers=point_list)

    def plot(
        self, ax: "GeoAxes", s: int = 10, **kwargs: Any
    ) -> "Artist":  # coverage: ignore
        """Plotting function. All arguments are passed to ax.scatter"""
        from cartopy.crs import PlateCarree

        return ax.scatter(  # type: ignore
            self.data.longitude,
            self.data.latitude,
            s=s,
            transform=PlateCarree(),
            **kwargs,
        )
