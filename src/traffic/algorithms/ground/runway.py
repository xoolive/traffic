from typing import TYPE_CHECKING, Iterator, Union

import pitot.geodesy as geo

import numpy as np
import pandas as pd
from shapely import LineString, MultiLineString, Point

from ...core import Flight
from ...core.structure import Airport

if TYPE_CHECKING:
    from cartes.osm import Overpass


class RunwayAlignment:
    """Iterates on all segments of trajectory matching a runway of the
    given airport.

    Example usage:

    .. code:: python

        >>> from traffic.data.samples import belevingsvlucht

    Count the number of segments aligned with a runway (take-off or landing):

    .. code:: python

        >>> sum(1 for _ in belevingsvlucht.aligned("EHAM", method="runway"))
        2

    Get timestamps associated with the first segment matching a runway:

    .. code:: python

        >>> segment = belevingsvlucht.next("aligned('EHAM', method='runway')")
        >>> f"{segment.start:%H:%M %Z}, {segment.stop:%H:%M %Z}"
        '20:17 UTC, 20:18 UTC'

    Get the minimum altitude for the aircraft landing:

    .. code:: python

        >>> segment.mean('altitude')  # Schiphol below the sea level
        -26.0

    """

    def __init__(self, airport: str | Airport):
        from ...data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if (
            self.airport is None
            or self.airport.runways is None
            or self.airport.runways.shape.is_empty
        ):
            raise RuntimeError("Airport or runway information missing")

    def apply(self, flight: Flight) -> Iterator[Flight]:
        if isinstance(self.airport.runways.shape, LineString):
            candidate_shapes = [
                LineString(list(flight.xy_time)).intersection(
                    self.airport.runways.shape.buffer(5e-4)
                )
            ]
        else:
            candidate_shapes = [
                LineString(list(flight.xy_time)).intersection(
                    on_runway.buffer(5e-4)
                )
                for on_runway in self.airport.runways.shape.geoms
            ]

        for intersection in candidate_shapes:
            if intersection.is_empty:
                continue
            if isinstance(intersection, LineString):
                (*_, start), *_, (*_, stop) = intersection.coords
                segment = flight.between(start, stop, strict=False)
                if segment is not None:
                    yield segment
            if isinstance(intersection, MultiLineString):
                (*_, start), *_, (*_, stop) = intersection.geoms[0].coords
                for chunk in intersection.geoms:
                    (*_, start_bak), *_, (*_, stop) = chunk.coords
                    if stop - start > 40:  # crossing runways and back
                        start = start_bak
                segment = flight.between(start, stop, strict=False)
                if segment is not None:
                    yield segment


class Deprecated:  # TODO
    def on_taxiway(
        self,
        flight: Flight,
        airport_or_taxiways: Union[str, pd.DataFrame, Airport, "Overpass"],
        *,
        tolerance: float = 15,
        max_dist: float = 85,
    ) -> Iterator["Flight"]:
        """
        Iterates on segments of trajectory matching a single runway label.
        """

        from ...data import airports

        if isinstance(airport_or_taxiways, str):
            airport_or_taxiways = airports[airport_or_taxiways]
        # This is obvious, but not for MyPy
        assert not isinstance(airport_or_taxiways, str)

        taxiways_ = (
            airport_or_taxiways.taxiway
            if isinstance(airport_or_taxiways, Airport)
            else airport_or_taxiways
        )

        # decompose with a function because MyPy is lost
        def taxi_df(taxiways_: Union["Overpass", pd.DataFrame]) -> pd.DataFrame:
            if isinstance(taxiways_, pd.DataFrame):
                return taxiways_
            if taxiways_.data is None:
                raise ValueError("No taxiway information")
            return taxiways_.data

        taxiways = (  # one entry per runway label
            taxi_df(taxiways_)
            .groupby("ref")
            .agg({"geometry": list})["geometry"]
            .apply(MultiLineString)
            .to_frame()
        )

        simplified_df = flight.simplify(tolerance=tolerance).data

        if simplified_df.shape[0] < 2:
            return

        previous_candidate = None
        first = simplified_df.iloc[0]
        for _, second in simplified_df.iloc[1:].iterrows():
            p1 = Point(first.longitude, first.latitude)
            p2 = Point(second.longitude, second.latitude)

            def extremities_dist(twy: MultiLineString) -> float:
                p1_proj = twy.interpolate(twy.project(p1))
                p2_proj = twy.interpolate(twy.project(p2))
                d1 = geo.distance(p1_proj.y, p1_proj.x, p1.y, p1.x)
                d2 = geo.distance(p2_proj.y, p2_proj.x, p2.y, p2.x)
                return d1 + d2  # type: ignore

            temp_ = taxiways.assign(dist=np.vectorize(extremities_dist))
            start, stop, ref, dist = (
                first.timestamp,
                second.timestamp,
                temp_.dist.idxmin(),
                temp_.dist.min(),
            )
            if dist < max_dist:
                candidate = flight.assign(taxiway=ref).between(start, stop)
                if previous_candidate is None:
                    previous_candidate = candidate

                else:
                    prev_ref = previous_candidate.taxiway_max
                    delta = start - previous_candidate.stop
                    if prev_ref == ref and delta < pd.Timedelta("1 min"):
                        previous_candidate = flight.assign(taxiway=ref).between(
                            previous_candidate.start, stop
                        )

                    else:
                        yield previous_candidate
                        previous_candidate = candidate

            first = second

        if previous_candidate is not None:
            yield previous_candidate
