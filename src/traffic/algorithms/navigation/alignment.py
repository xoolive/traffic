import logging
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Sequence,
)

import numpy as np
import pandas as pd

from ...core import Flight, FlightPlan
from ...core.mixins import PointMixin

if TYPE_CHECKING:
    from ...data.basic.navaid import Navaids

_log = logging.getLogger(__name__)


class AlignmentBase(Protocol):
    def apply(self, flight: Flight) -> Iterator[Flight]: ...


class BeaconTrackBearingAlignment:
    """Iterates on segments of trajectories aligned with one of the given
    navigational beacons passed in parameter.

    The name of the navigational beacon is assigned in a new column
    `navaid`.

    """

    points: Sequence[PointMixin]

    def __init__(
        self,
        points: str | PointMixin | Iterable[PointMixin] | FlightPlan,
        angle_precision: int = 1,
        time_precision: str = "2 min",
        min_time: str = "30s",
        min_distance: int = 80,
    ) -> None:
        self.angle_precision = angle_precision
        self.time_precision = time_precision
        self.min_time = min_time
        self.min_distance = min_distance

        if isinstance(points, str):
            from ...data import navaids

            navaid = navaids[points]
            if navaid is None:
                _log.warning(f"Navaid {points} unknown")
                self.points = []
            else:
                self.points = [navaid]
        elif isinstance(points, PointMixin):
            self.points = [points]
        elif isinstance(points, FlightPlan):
            self.points = points.all_points
        else:
            self.points = list(points)

    def apply(self, flight: Flight) -> Iterator[Flight]:
        for navpoint in self.points:
            tentative = (
                flight.distance(navpoint)
                .bearing(navpoint)
                .assign(
                    shift=lambda df: df.distance
                    * (np.radians(df.bearing - df.track).abs()),
                    delta=lambda df: (df.bearing - df.track).abs(),
                )
                .query(f"delta < {self.angle_precision} and distance < 500")
            )
            if tentative is not None:
                for segment in tentative.split(self.time_precision):
                    if (
                        segment.longer_than(self.min_time)
                        and segment.min("distance") < self.min_distance
                    ):
                        yield segment.assign(navaid=navpoint.name)


class ExtractPoint:
    def compute_navpoints(
        self,
        flight: Flight,
        navaids: Optional["Navaids"] = None,
        buffer: float = 0.1,
    ) -> None | pd.DataFrame:
        """This functions recomputes the most probable alignments on
        navigational points on the trajectory.

        By default, all navaids of the default database are considered,
        but limited to a buffered bounding box around the trajectory.

        Once computed, the following Altair snippet may be useful to
        display the trajectory as a succession of segments:

        .. code:: python

            import altair as alt

            df = flight.compute_navpoints()

            segments = (
                alt.Chart(df.drop(columns="duration")).encode(
                    alt.X("start", title=None),
                    alt.X2("stop"),
                    alt.Y("navaid", sort="x", title=None),
                    alt.Color("type", title="Navigational point"),
                    alt.Tooltip(["navaid", "distance", "shift_mean"]),
                )
                .mark_bar(size=10)
                .configure_legend(
                    orient="bottom",
                    labelFontSize=14, titleFontSize=14, labelFont="Ubuntu"
                )
                .configure_axis(labelFontSize=14, labelFont="Ubuntu")
            )

        """

        if navaids is None:
            from ...data import navaids as default_navaids

            navaids = default_navaids

        navaids_ = navaids.extent(flight, buffer=buffer)
        if navaids_ is None:
            return None
        navaids_ = navaids_.drop_duplicates("name")
        all_points = list(navaids_)

        def all_aligned_segments(traj: "Flight") -> pd.DataFrame:
            return pd.DataFrame.from_records(
                list(
                    {
                        "start": segment.start,
                        "stop": segment.stop,
                        "duration": segment.duration,
                        "navaid": segment.max("navaid"),
                        "distance": segment.min("distance"),
                        "shift_mean": segment.shift_mean,
                        "shift_meanp": segment.shift_mean + 0.02,
                    }
                    for segment in traj.aligned_on_navpoint(all_points)
                )
            ).sort_values("start")

        def groupby_intervals(table: pd.DataFrame) -> Iterator[pd.DataFrame]:
            if table.shape[0] == 0:
                return
            table = table.sort_values("start")
            # take as much as you can
            sweeping_line = table.query("stop <= stop.iloc[0]")
            # try to push the stop line: which intervals overlap the stop line?
            additional = table.query(
                "start <= @sweeping_line.stop.max() < stop"
            )

            while additional.shape[0] > 0:
                sweeping_line = table.query("stop <= @additional.stop.max()")
                additional = table.query(
                    "start <= @sweeping_line.stop.max() < stop"
                )

            yield sweeping_line
            yield from groupby_intervals(
                table.query("start > @sweeping_line.stop.max()")
            )

        def most_probable_navpoints(traj: "Flight") -> Iterator[pd.DataFrame]:
            table = all_aligned_segments(traj)
            for block in groupby_intervals(table):
                d_max = block.eval("duration.max()")
                t_threshold = d_max - pd.Timedelta("30s")  # noqa: F841
                yield (
                    block.sort_values("shift_mean")
                    .query("duration >= @t_threshold")
                    .head(1)
                )

        return pd.concat(list(most_probable_navpoints(flight))).merge(
            navaids_.data, left_on="navaid", right_on="name"
        )
