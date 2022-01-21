from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import geopandas as gpd
from cartes.utils.cache import cached_property

import pandas as pd
from shapely.geometry import base, shape
from shapely.ops import orient, unary_union

from ....core.airspace import Airspace, ExtrudedPolygon, unary_union_with_alt
from ....core.mixins import DataFrameMixin

# https://www.nm.eurocontrol.int/HELP/Airspaces.html


def _re_match_ignorecase(x: str, y: str) -> Optional[re.Match[str]]:
    return re.match(x, y, re.IGNORECASE)


class NMAirspaceParser(DataFrameMixin):

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

    @cached_property
    def __geo_interface__(self) -> Any:
        return self.dissolve().__geo_interface__

    def dissolve(self) -> gpd.GeoDataFrame:
        columns = ["designator", "type", "upper", "lower"]
        name_table = self.data[["designator", "name"]].drop_duplicates()

        return gpd.GeoDataFrame(
            self.data.groupby(columns)
            .agg(dict(geometry=unary_union))
            .reset_index()
            .merge(name_table)
        )

    def __getitem__(self, name: str) -> None | Airspace:
        subset = self.query(f'designator == "{name}"')
        if subset is None:
            return None

        return Airspace(
            elements=unary_union_with_alt(
                [
                    ExtrudedPolygon(line.geometry, line.lower, line.upper)
                    for _, line in subset.data.iterrows()
                ]
            ),
            name=subset.data.name.max(),
            type_=subset.data["type"].max(),
            designator=subset.data.designator.max(),
        )

    def __iter__(self) -> Iterator[Airspace]:
        for _, subset in self.groupby("designator"):
            yield Airspace(
                elements=unary_union_with_alt(
                    [
                        ExtrudedPolygon(line.geometry, line.lower, line.upper)
                        for _, line in subset.iterrows()
                    ]
                ),
                name=subset.name.max(),
                type_=subset["type"].max(),
                designator=subset.designator.max(),
            )

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
        )

        self.consolidate()

    def read_are(self, filename: Path) -> None:
        logging.info(f"Reading ARE file {filename}")
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
        logging.info(f"Reading SLS file {filename}")
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
        logging.info(f"Reading SPC file {filename}")
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

    @lru_cache()
    def _ipython_key_completions_(self) -> Set[str]:
        return set(self.data.designator)

    def consolidate(self) -> gpd.GeoDataFrame:
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
        self.data = gpd.GeoDataFrame(new_data)
        return self.data
