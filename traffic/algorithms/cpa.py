import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Iterator, Set, Tuple, Union, cast

import pandas as pd
import pyproj
from cartopy import crs
from tqdm.autonotebook import tqdm
from traffic.core.mixins import DataFrameMixin

from ..core import Flight, Traffic


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
        def _minimum_column(df):
            x = df.loc[df[column].idxmin()]
            if len(x.shape) > 1:
                x = x.iloc[0]
            return x

        return self.__class__(
            self.groupby(["flight_id_x", "flight_id_y"])
            .apply(_minimum_column)
            .drop(columns=["flight_id_x", "flight_id_y"])
            .reset_index()
        )

    def _ipython_key_completions_(self) -> Set[str]:
        return self.flight_ids()

    def flight_ids(self) -> Set[str]:
        return {*set(self.data.flight_id_x), *set(self.data.flight_id_y)}

    def __getitem__(self, index: str) -> "CPA":
        df = self.data
        if not isinstance(index, str) or index in df.columns:
            return df[index]

        return self.__class__(
            df.query("flight_id_y == @index").append(
                df.query("flight_id_x == @index").rename(
                    columns={
                        "latitude_x": "latitude_y",
                        "latitude_y": "latitude_x",
                        "longitude_x": "longitude_y",
                        "longitude_y": "longitude_x",
                        "altitude_x": "altitude_y",
                        "altitude_y": "altitude_x",
                        "flight_id_x": "flight_id_y",
                        "flight_id_y": "flight_id_x",
                    }
                ),
                sort=False,
            )
        )

    def _repr_html_(self):
        try:
            return self.data.style.background_gradient(
                subset=["aggregated"], cmap="bwr_r", low=.9, high=1.
            )._repr_html_()
        except Exception:
            return super()._repr_html_()


def closest_point_of_approach(
    traffic: Traffic,
    lateral_separation: float,
    vertical_separation: float,
    projection: Union[pyproj.Proj, crs.Projection, None] = None,
    round_t: str = "d",
    max_workers: int = 4,
) -> CPA:
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

    def yield_pairs(t_chunk: Traffic):
        """
        This function yields all pairs of possible candidates for a CPA
        calculation.
        """

        # combinations types Iterator[Tuple[T, ...]]
        for first, second in cast(
            Iterator[Tuple[Flight, Flight]], combinations(t_chunk, 2)
        ):
            # cast are necessary because of the lru_cache × property bug
            if (
                cast(pd.Timestamp, first.start)
                > cast(pd.Timestamp, second.stop)
            ) or (
                cast(pd.Timestamp, second.start)
                > cast(pd.Timestamp, first.stop)
            ):
                # Flights must fly at the same time
                continue
            if (
                first.min("altitude")
                > second.max("altitude") + vertical_separation
            ):
                # Bounding boxes in altitude must cross
                continue
            if (
                second.min("altitude")
                > first.max("altitude") + vertical_separation
            ):
                # Bounding boxes in altitude must cross
                continue
            if first.min("x") > second.max("x") + lateral_separation:
                # Bounding boxes in x must cross
                continue
            if second.min("x") > first.max("x") + lateral_separation:
                # Bounding boxes in x must cross
                continue
            if first.min("y") > second.max("y") + lateral_separation:
                # Bounding boxes in y must cross
                continue
            if second.min("y") > first.max("y") + lateral_separation:
                # Bounding boxes in y must cross
                continue

            # Next step is to check the 2D footprint of the trajectories
            # intersect. Before computing the intersection we bufferize the
            # trajectories by half the requested separation.

            first_shape = first.project_shape(projection)
            second_shape = second.project_shape(projection)
            if first_shape is None or second_shape is None:
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
                # TODO submit(Flight.distance, first, second)
                executor.submit(first.distance, second): (
                    first.flight_id,
                    second.flight_id,
                )
                for (first, second) in yield_pairs(Traffic(t_chunk))
            }

            for future in as_completed(tasks):
                cumul.append(future.result())

    return CPA(pd.concat(cumul, sort=False))
