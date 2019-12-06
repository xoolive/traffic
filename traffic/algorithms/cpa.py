import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Iterator, Optional, Set, Tuple, Union

import pandas as pd
import pyproj
from cartopy import crs
from tqdm.autonotebook import tqdm

from ..core import Flight
from ..core.mixins import DataFrameMixin

if TYPE_CHECKING:
    from ..core import Traffic


def combinations(
    t: "Traffic", lateral_separation: float, vertical_separation: float
) -> Iterator[Tuple["Flight", "Flight"]]:
    for flight in tqdm(t, desc="Combinations", leave=False):

        t_ = t.query(f'icao24 != "{flight.icao24}"')
        if t_ is None:
            continue

        clipped = t_.query(
            f'x >= {flight.min("x")} - {lateral_separation} and '
            f'x <= {flight.max("x")} + {lateral_separation} and '
            f'y >= {flight.min("y")} - {lateral_separation} and '
            f'y <= {flight.max("y")} + {lateral_separation} and '
            f'altitude >= {flight.min("altitude")} - {vertical_separation} and '
            f'altitude <= {flight.max("altitude")} + {vertical_separation} and '
            f'timestamp <= "{flight.stop}" and '
            f'timestamp >= "{flight.start}" '
        )
        if clipped is None:
            continue

        for second in clipped:
            yield flight, second


class CPA(DataFrameMixin):
    def aggregate(
        self, lateral_separation: float = 5, vertical_separation: float = 1000
    ) -> "CPA":
        return (
            self.assign(
                tmp_lateral=lambda df: df.lateral / lateral_separation,
                tmp_vertical=lambda df: df.vertical / vertical_separation,
            )
            .assign(
                aggregated=lambda df: df[["tmp_lateral", "tmp_vertical"]].max(
                    axis=1
                )
            )
            .drop(columns=["tmp_lateral", "tmp_vertical"])
        )

    def min(self, column: str) -> "CPA":
        return self.__class__(
            self.sort_values(column).groupby(["icao24_x", "icao24_y"]).first()
        )

    def _ipython_key_completions_(self) -> Set[str]:
        return self.flight_ids()

    def flight_ids(self) -> Set[str]:
        return {*set(self.data.flight_id_x), *set(self.data.flight_id_y)}

    def __getitem__(self, index: str) -> "CPA":
        df = self.data
        if not isinstance(index, str) or index in df.columns:
            return df[index]

        rename_cols = {
            "latitude_x": "latitude_y",
            "latitude_y": "latitude_x",
            "longitude_x": "longitude_y",
            "longitude_y": "longitude_x",
            "altitude_x": "altitude_y",
            "altitude_y": "altitude_x",
            "icao24_x": "icao24_y",
            "icao24_y": "icao24_x",
            "callsign_x": "callsign_y",
            "callsign_y": "callsign_x",
        }

        if "flight_id_x" in self.data.columns:
            rename_cols["flight_id_x"] = "flight_id_y"
            rename_cols["flight_id_y"] = "flight_id_x"

            res = df.query("flight_id_y == @index").append(
                df.query("flight_id_x == @index").rename(columns=rename_cols),
                sort=False,
            )

            if res.shape[0] > 0:
                return self.__class__(res)

        res = df.query("icao24_y == @index").append(
            df.query("icao24_x == @index").rename(columns=rename_cols),
            sort=False,
        )

        return self.__class__(res)

    def _repr_html_(self):
        try:
            return self.data.style.background_gradient(
                subset=["aggregated"], cmap="bwr_r", low=0.9, high=1.0
            )._repr_html_()
        except Exception:
            return super()._repr_html_()


def closest_point_of_approach(
    traffic: "Traffic",
    lateral_separation: float,
    vertical_separation: float,
    projection: Union[pyproj.Proj, crs.Projection, None] = None,
    round_t: str = "d",
    max_workers: int = 4,
) -> Optional[CPA]:
    """
    Computes a CPA dataframe for all pairs of trajectories candidates for
    being separated by less than lateral_separation in vertical_separation.

    In order to be computed efficiently, the method needs the following
    parameters:

    - projection: a first filtering is applied on the bounding boxes of
    trajectories, expressed in meters. You need to provide a decent
    projection able to approximate distances by Euclide formula.
    By default, EuroPP() projection is considered, but a non explicit
    argument will raise a warning.

    - round_t: an additional column will be added in the DataFrame to group
    trajectories by relevant time frames. Distance computations will be
    considered only between trajectories flown in the same time frame.
    By default, the 'd' pandas freq parameter is considered, to group
    trajectories by day, but other ways of splitting ('h') may be more
    relevant and impact performance.

    - max_workers: distance computations are spread over a given number of
    processors.

    """

    if projection is None:
        logging.warn("Defaulting to projection EuroPP()")
        projection = crs.EuroPP()

    if isinstance(projection, crs.Projection):
        projection = pyproj.Proj(projection.proj4_init)

    def yield_pairs(t_chunk: "Traffic"):
        """
        This function yields all pairs of possible candidates for a CPA
        calculation.
        """

        for first, second in combinations(
            t_chunk,
            lateral_separation=lateral_separation,
            vertical_separation=vertical_separation,
        ):
            # Next step is to check the 2D footprint of the trajectories
            # intersect. Before computing the intersection we bufferize the
            # trajectories by half the requested separation.

            first_shape = first.project_shape(projection)
            second_shape = second.project_shape(projection)
            if first_shape is None or second_shape is None:
                continue
            if not first_shape.is_valid or not second_shape.is_valid:
                continue

            first_shape = first_shape.simplify(1e3).buffer(
                lateral_separation / 2
            )
            second_shape = first_shape.simplify(1e3).buffer(
                lateral_separation / 2
            )

            if first_shape.intersects(second_shape):
                yield first, second

    t_xyt = (
        traffic.airborne()
        .compute_xy(projection)
        .assign(round_t=lambda df: df.timestamp.dt.round(round_t))
    )

    cumul = list()

    # Multiprocessing is implemented on each timerange slot only.
    # TODO: it would probably be more efficient to multiprocess over each
    # t_chunk rather than multiprocess the distance computation.

    for _, t_chunk in tqdm(
        t_xyt.groupby("round_t"), total=len(set(t_xyt.data.round_t))
    ):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = {
                executor.submit(Flight.distance, first, second): (
                    first.icao24,
                    second.icao24,
                )
                for (first, second) in yield_pairs(traffic.__class__(t_chunk))
            }

            for future in as_completed(tasks):
                result = future.result()
                if result is not None:
                    cumul.append(result)

    if len(cumul) == 0:
        return None

    return CPA(pd.concat(cumul, sort=False))
