from typing import (
    TYPE_CHECKING,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import pandas as pd

from ...core.flight import Flight
from ...core.mixins import PointMixin

if TYPE_CHECKING:
    from ...data.basic.navaid import Navaids


@runtime_checkable
class FlightPlanBase(Protocol):
    def infer(self, flight: Flight) -> None | pd.DataFrame: ...


class FlightPlanInference:
    """This functions recomputes the most probable alignments on navigational
    points on the trajectory.

    :param navaid: By default, all navaids of the default database are
       considered, but limited to a buffered bounding box around the trajectory.

       It should be good practice to limit the size of the navaid dataset: the
       smaller the data, the faster the computation will be.

    >>> from traffic.data import navaids
    >>> from traffic.data.samples import savan
    >>> flight = savan["SAVAN01"]
    >>> vor = navaids.query("type == 'VOR'")
    >>> df = flight.infer_flightplan(vor)
    >>> for _, line in df.iterrows():
    ...     print(f"aligned on {line.navaid} ({line.type}) for {line.duration}")
    aligned on CFA (VOR) for 0 days 00:21:19
    aligned on POI (VOR) for 0 days 00:15:42
    aligned on CAN (VOR) for 0 days 00:14:38
    aligned on DPE (VOR) for 0 days 00:16:19
    aligned on CHW (VOR) for 0 days 00:15:45
    aligned on BRY (VOR) for 0 days 00:19:27

    For more insights about this example:
    :ref:`Calibration flights with SAVAN trajectories`

    Once computed, the following Altair snippet may be useful to display the
    trajectory as a succession of segments:

    .. code:: python

        import altair as alt

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

    def __init__(
        self,
        navaids: Optional["Navaids"] = None,
        buffer: float = 0.1,
    ):
        from ...data import navaids as default_navaids

        self.navaids = navaids if navaids is not None else default_navaids
        self.buffer = buffer

    def all_aligned_segments(
        self, traj: "Flight", all_points: Sequence[PointMixin]
    ) -> pd.DataFrame:
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
                for segment in traj.aligned(all_points)
            )
        ).sort_values("start")

    def groupby_intervals(self, table: pd.DataFrame) -> Iterator[pd.DataFrame]:
        if table.shape[0] == 0:
            return
        table = table.sort_values("start")
        # take as much as you can
        sweeping_line = table.query("stop <= stop.iloc[0]")
        # try to push the stop line: which intervals overlap the stop line?
        additional = table.query("start <= @sweeping_line.stop.max() < stop")

        while additional.shape[0] > 0:
            sweeping_line = table.query("stop <= @additional.stop.max()")
            additional = table.query(
                "start <= @sweeping_line.stop.max() < stop"
            )

        yield sweeping_line
        yield from self.groupby_intervals(
            table.query("start > @sweeping_line.stop.max()")
        )

    def most_probable_navpoints(
        self, traj: "Flight", all_points: Sequence[PointMixin]
    ) -> Iterator[pd.DataFrame]:
        table = self.all_aligned_segments(traj, all_points)
        for block in self.groupby_intervals(table):
            d_max = block.eval("duration.max()")
            t_threshold = d_max - pd.Timedelta("30s")  # noqa: F841
            yield (
                block.sort_values("shift_mean")
                .query("duration >= @t_threshold")
                .head(1)
            )

    def infer(self, flight: Flight) -> None | pd.DataFrame:
        navaids_ = self.navaids.extent(flight, buffer=self.buffer)
        if navaids_ is None:
            return None
        navaids_ = navaids_.drop_duplicates("name")
        all_points = list(navaids_)

        return pd.concat(
            list(self.most_probable_navpoints(flight, all_points))
        ).merge(navaids_.data, left_on="navaid", right_on="name")
