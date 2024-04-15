from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar

import geopandas as gpd

import pandas as pd
from shapely.geometry import base, shape
from shapely.ops import orient

from ....core.airspace import Airspaces

A = TypeVar("A", bound="NMAirspaceParser")

_log = logging.getLogger(__name__)


class NMAirspaceParser(Airspaces):
    # https://www.nm.eurocontrol.int/HELP/Airspaces.html

    nm_path: Optional[Path] = None

    def __init__(
        self,
        data: pd.DataFrame | None,
        config_file: Path | None = None,
    ) -> None:
        super().__init__(data, config_file)

        if self.data is not None:
            return

        self.config_file = config_file

        self.polygons: Dict[str, base.BaseGeometry] = {}
        self.elements_list: list[dict[str, Any]] = list()
        self.spc_list: list[dict[str, Any]] = list()

        self.init_cache()

    def update_path(self, path: Path) -> None:
        self.nm_path = path
        self.init_cache()

    def init_cache(self) -> None:
        msg = f"Edit file {self.config_file} with NM directory"

        if self.nm_path is None:
            raise RuntimeError(msg)

        are_file = next(self.nm_path.glob("Sectors_*.are"), None)
        if are_file is None:
            raise RuntimeError(f"No Sectors_*.are file found in {self.nm_path}")
        self.read_are(are_file)

        sls_file = next(self.nm_path.glob("Sectors_*.sls"), None)
        if sls_file is None:
            raise RuntimeError(f"No Sectors_*.sls file found in {self.nm_path}")
        self.read_sls(sls_file)

        spc_file = next(self.nm_path.glob("Sectors_*.spc"), None)
        if spc_file is None:
            raise RuntimeError(f"No Sectors_*.spc file found in {self.nm_path}")
        # self.read_spc(spc_file)

        self.data = gpd.GeoDataFrame(
            pd.DataFrame.from_records(self.read_spc(spc_file))
            .merge(
                gpd.GeoDataFrame.from_records(self.elements_list).rename(
                    columns=dict(designator="component")
                ),
                how="left",
                on="component",
            )
            .query("designator != component or geometry.notnull()")
            .assign(upper=lambda df: df.upper.replace(999, float("inf")))
            .set_geometry("geometry")
        )

    def read_are(self, filename: Path) -> None:
        _log.info(f"Reading ARE file {filename}")
        nb_points = 0
        area_coords: List[Tuple[float, float]] = list()
        name = None
        with filename.open("r") as f:
            for line in f.readlines():
                if nb_points == 0:
                    if name is not None:
                        geometry = {
                            "type": "Polygon",
                            "coordinates": [area_coords],
                        }
                        self.polygons[name] = orient(shape(geometry), -1)

                    area_coords.clear()
                    nb, *_, name = line.split()
                    nb_points = int(nb)
                else:
                    lat, lon = line.split()
                    area_coords.append((float(lon) / 60, float(lat) / 60))
                    nb_points -= 1

            if name is not None:
                geometry = {"type": "Polygon", "coordinates": [area_coords]}
                self.polygons[name] = orient(shape(geometry), -1)

    def read_sls(self, filename: Path) -> None:
        _log.info(f"Reading SLS file {filename}")
        with filename.open("r") as fh:
            for line in fh.readlines():
                name, _, polygon, lower, upper = line.split()
                self.elements_list.append(
                    dict(
                        geometry=self.polygons[polygon],
                        designator=name,
                        upper=float(upper),
                        lower=float(lower),
                    )
                )

    def read_spc(self, filename: Path) -> Iterator[dict[str, Any]]:
        _log.info(f"Reading SPC file {filename}")
        with open(filename, "r") as fh:
            for line in fh.readlines():
                letter, component, *after = line.split(";")
                if letter == "A":
                    description = after[0]
                    type_ = after[1]
                    name = component
                elif letter == "S":
                    yield dict(
                        designator=name,
                        component=component,
                        name=description,
                        type=type_,
                    )
                    yield dict(
                        designator=component,
                        component=component,
                        type=after[0].strip(),
                    )

    def consolidate(self: "A") -> "A":
        def consolidate_rec(df: pd.DataFrame) -> pd.DataFrame:
            if df.geometry.notnull().all():
                return df
            return consolidate_rec(
                pd.concat(
                    [
                        df.query("geometry.notnull()"),
                        df.query("geometry.isnull()")
                        .drop(columns=["geometry", "upper", "lower"])
                        .drop_duplicates()
                        .merge(
                            df.drop(columns=["component", "name", "type"]),
                            left_on="component",
                            right_on="designator",
                            suffixes=["", "_2"],
                        )
                        .drop(columns=["designator_2"]),
                    ]
                )
            )

        new_data = pd.DataFrame(self.data).pipe(consolidate_rec)

        return self.__class__(gpd.GeoDataFrame(new_data))
