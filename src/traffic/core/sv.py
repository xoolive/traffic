from typing import TYPE_CHECKING, Any, Iterator, Set

import pandas as pd

from .flight import Position
from .mixins import GeographyMixin

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from ipyleaflet import MarkerCluster as LeafletMarkerCluster
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

    def leaflet(sv, **kwargs: Any) -> "LeafletMarkerCluster":
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

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


def patch_leaflet() -> None:
    from ..visualize.leaflet import statevector_leaflet

    StateVectors.leaflet = statevector_leaflet  # type: ignore


try:
    patch_leaflet()
except Exception:
    pass
