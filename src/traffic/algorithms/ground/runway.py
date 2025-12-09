import math
from typing import TYPE_CHECKING, Iterator, Union

import pitot.geodesy as geo

import numpy as np
import pandas as pd
import pyproj
from shapely import LineString, MultiLineString, Point, Polygon
from shapely.affinity import rotate, translate
from shapely.geometry import box
from shapely.ops import transform
from shapely.prepared import prep
from shapely.validation import make_valid

from ...core import Flight
from ...core.structure import Airport

if TYPE_CHECKING:
    from cartes.osm import Overpass


class RunwayAlignment:
    """Find trajectory segments aligned with airport runways.

    Methodology (mirrors :meth:`Flight.aligned_on_runway` from the other
    project):
    - Build a flat-ended rectangle for each runway centerline using its true
      length and the provided full width (meters).
    - Test each trajectory point for inclusion in the runway polygon
      (prepared geometry for speed).
    - Keep contiguous runs with at least ``min_points``; then drop runs shorter
      than ``min_duration``.
    - Merge overlapping/adjacent runs per runway within ``overlap_tolerance``.
    - Yield flight segments annotated with ``rwy`` (e.g., ``"04/22"``).

    Parameters
    ----------
    airport : str | Airport
        ICAO/IATA code or airport instance providing runway geometry.
    width_m : float, default 60.0
        Full runway width in meters used to build rectangular footprints.
    min_points : int, default 3
        Minimum consecutive trajectory points inside the runway polygon.
    min_duration : timedelta | str, default "8s"
        Minimum segment duration after point filtering.
    overlap_tolerance : timedelta | str, default "1s"
        Maximum gap when merging overlapping/adjacent runs per runway.

    Notes
    -----
    Both the point-count and duration thresholds must be satisfied for a
    segment to be yielded.
    """

    def __init__(
        self,
        airport: str | Airport,
        *,
        width_m: float = 60.0,
        min_points: int = 3,
        min_duration: pd.Timedelta | str = "5s",
        overlap_tolerance: pd.Timedelta | str = "1s",
    ):
        from ...data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        self.width_m = width_m
        self.min_points = min_points
        self.min_duration = pd.to_timedelta(min_duration)
        self.overlap_tolerance = pd.to_timedelta(overlap_tolerance)
        if (
            self.airport is None
            or self.airport.runways is None
            or self.airport.runways.shape.is_empty
        ):
            raise RuntimeError("Airport or runway information missing")

    def _runway_polygon(self, thr0, thr1, width_m: float) -> Polygon:
        line = LineString(
            [(thr0.longitude, thr0.latitude), (thr1.longitude, thr1.latitude)]
        )
        centroid = line.centroid
        proj_local = pyproj.Proj(
            proj="aeqd", lat_0=centroid.y, lon_0=centroid.x
        )
        to_local = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), proj_local, always_xy=True
        )
        to_wgs84 = pyproj.Transformer.from_proj(
            proj_local, pyproj.Proj("epsg:4326"), always_xy=True
        )

        line_local = transform(to_local.transform, line)
        if line_local.length == 0:
            polygon = line
        else:
            x0, y0 = line_local.coords[0]
            x1, y1 = line_local.coords[-1]
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
            length_m = line_local.length

            rect_local = box(
                -length_m / 2, -width_m / 2, length_m / 2, width_m / 2
            )
            rect_rotated = rotate(
                rect_local, angle, origin=(0, 0), use_radians=False
            )
            center = line_local.interpolate(0.5, normalized=True)
            rect_shifted = translate(
                rect_rotated, xoff=center.x, yoff=center.y
            )
            polygon = transform(to_wgs84.transform, rect_shifted)

        return polygon if polygon.is_valid else make_valid(polygon)

    def _contiguous_runs(self, mask: np.ndarray) -> list[tuple[int, int]]:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []
        runs: list[tuple[int, int]] = []
        start, prev = indices[0], indices[0]
        for current in indices[1:]:
            if current == prev + 1:
                prev = current
                continue
            runs.append((start, prev))
            start = prev = current
        runs.append((start, prev))
        return runs

    def apply(self, flight: Flight) -> Iterator[Flight]:
        """Yield flight segments aligned with any runway of the configured
        airport."""
        data = (
            flight.data.query("longitude.notnull() and latitude.notnull()")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if data.shape[0] == 0:
            return None

        coords = data[["longitude", "latitude"]].to_numpy()
        timestamps = data["timestamp"]

        found_segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []

        for thr0, thr1 in getattr(self.airport.runways, "_runways", []):
            rwy_polygon = self._runway_polygon(thr0, thr1, self.width_m)
            if rwy_polygon.is_empty:
                continue

            rwy_name = f"{thr0.name}/{thr1.name}"
            prepared = prep(rwy_polygon)
            mask = np.fromiter(
                (prepared.intersects(Point(lon, lat)) for lon, lat in coords),
                dtype=bool,
                count=len(coords),
            )

            for start_idx, stop_idx in self._contiguous_runs(mask):
                if stop_idx - start_idx + 1 < self.min_points:
                    continue
                start_ts = timestamps.iloc[start_idx]
                stop_ts = timestamps.iloc[stop_idx]
                found_segments.append((start_ts, stop_ts, rwy_name))

        if len(found_segments) == 0:
            return None

        tolerance = self.overlap_tolerance
        min_duration = self.min_duration
        per_rwy: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}

        for start_ts, stop_ts, rwy_name in found_segments:
            if stop_ts - start_ts < min_duration:
                continue
            per_rwy.setdefault(rwy_name, []).append((start_ts, stop_ts))

        ordered_segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []

        for rwy_name, runs in per_rwy.items():
            cleaned: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for start_ts, stop_ts in sorted(runs, key=lambda elt: elt[0]):
                if not cleaned:
                    cleaned.append((start_ts, stop_ts))
                    continue

                prev_start, prev_stop = cleaned[-1]
                if start_ts <= prev_stop + tolerance:
                    prev_dur = prev_stop - prev_start
                    cur_dur = stop_ts - start_ts
                    if cur_dur > prev_dur:
                        cleaned[-1] = (start_ts, stop_ts)
                    continue

                cleaned.append((start_ts, stop_ts))

            for start_ts, stop_ts in cleaned:
                ordered_segments.append((start_ts, stop_ts, rwy_name))

        for start_ts, stop_ts, rwy_name in sorted(
            ordered_segments, key=lambda elt: elt[0]
        ):
            segment = flight.between(start_ts, stop_ts, strict=False)
            if segment is None:
                continue
            yield segment.assign(rwy=rwy_name)


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
