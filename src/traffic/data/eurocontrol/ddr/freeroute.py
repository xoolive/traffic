import re
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Set, Tuple

import geopandas as gpd

import pandas as pd
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

from ....data import nm_navaids
from .airspaces import NMAirspaceParser


def parse_coordinates(elt: str) -> Tuple[float, float]:
    pattern = r"([N,S])(\d{4}|\d{6})(.\d*)?([E,W])(\d{5}|\d{7})(.\d*)?$"
    x = re.match(pattern, elt)
    assert x is not None, elt
    lat_, lat_sign = x.group(2), 1 if x.group(1) == "N" else -1
    lon_, lon_sign = x.group(5), 1 if x.group(4) == "E" else -1
    lat_ = lat_.ljust(6, "0")
    lon_ = lon_.ljust(7, "0")

    lat = lat_sign * (
        int(lat_[:2]) + int(lat_[2:4]) / 60 + int(lat_[4:]) / 3600
    )
    lon = lon_sign * (
        int(lon_[:3]) + int(lon_[3:5]) / 60 + int(lon_[5:]) / 3600
    )

    return (lat, lon)


class NMFreeRouteParser(NMAirspaceParser):
    def init_cache(self) -> None:
        msg = f"Edit file {self.config_file} with NM directory"

        if self.nm_path is None:
            raise RuntimeError(msg)

        are_file = next(self.nm_path.glob("Free_Route_*.are"), None)
        if are_file is None:
            raise RuntimeError(
                f"No Free_Route_*.are file found in {self.nm_path}"
            )
        self.read_are(are_file)

        sls_file = next(self.nm_path.glob("Free_Route_*.sls"), None)
        if sls_file is None:
            raise RuntimeError(
                f"No Free_Route_*.sls file found in {self.nm_path}"
            )
        self.read_sls(sls_file)

        data = gpd.GeoDataFrame.from_records(self.elements_list)
        data = data.set_geometry("geometry")
        if "name" not in data.columns:
            data = data.assign(name="")
        self.fra = self.data = data.assign(type="FRA")

        frp_file = next(self.nm_path.glob("Free_Route_*.frp"), None)
        if frp_file is None:
            raise RuntimeError(
                f"No Free_Route_*.frp file found in {self.nm_path}"
            )
        self.read_frp(frp_file)

    def read_frp(self, filename: Path) -> None:
        area = unary_union(self.fra.geometry)
        west, south, east, north = area.bounds

        subset = nm_navaids.extent((west, east, south, north))
        assert subset is not None
        coords = subset.data[["longitude", "latitude"]].values
        europoints = subset.data.merge(
            pd.DataFrame(
                [
                    list(x.coords[0])
                    for x in area.intersection(MultiPoint(coords)).geoms
                ],
                columns=["longitude", "latitude"],
            )
        )

        df = pd.read_csv(StringIO(filename.read_text()), header=None)
        df_ = (
            df[0]
            .str.replace(r"\s+", " ", regex=True)
            .str.split(" ", expand=True)
            .rename(columns={0: "FRA", 1: "type", 2: "name"})
        )

        a = (
            df_.query('type in ["AD", "A", "D"]')
            .dropna(axis=1, how="all")
            .iloc[:, 3:]
            .fillna("")
            .sum(axis=1)
            .str.replace(r"(\w{4})", r"\1,", regex=True)
            .str[:-1]
            .str.split(",")
        )

        tab = (
            df_.query('type not in ["AD", "A", "D"]')
            .dropna(axis=1, how="all")
            .rename(columns={3: "latitude", 4: "longitude"})
        )

        # Part 1: When coordinates are in the file, decode them
        coords = (
            tab.query("latitude.notnull()")[["latitude", "longitude"]]
            .sum(axis=1)
            .apply(parse_coordinates)
        )
        decode_coords = tab.query("latitude.notnull()").assign(
            latitude=coords.str[0], longitude=coords.str[1]
        )

        # Part 2: Propagate decoded coordinates (avoid slight inconsistencies)
        propagate_coords = (
            tab.query("latitude.isnull() and name in @decode_coords.name")
            .drop(columns=["latitude", "longitude"])
            .merge(
                decode_coords[
                    ["name", "latitude", "longitude"]
                ].drop_duplicates(),
                on="name",
            )
        )

        # Part 3: Unknown coordinates

        unknown_coords = (
            tab.query("latitude.isnull() and name not in @decode_coords.name")
            .drop(columns=["latitude", "longitude"])
            .merge(europoints.drop(columns=["type", "description"]), on="name")
        )

        # Part 4: Airport connections

        airport_coords = pd.concat(
            [
                df_.query('type in ["AD", "A", "D"]').iloc[:, :3],
                a.rename("airport"),
            ],
            axis=1,
        )

        propagate_airports = airport_coords.merge(
            decode_coords[["name", "latitude", "longitude"]].drop_duplicates(),
            on=["name"],
        ).explode("airport")

        unknown_airports = (
            airport_coords.query("name not in @propagate_airports.name").merge(
                europoints.drop(columns=["type", "description"]), on="name"
            )
        ).explode("airport")

        self.frp = self.points = pd.concat(
            [
                decode_coords,
                propagate_coords,
                unknown_coords,
                propagate_airports,
                unknown_airports,
            ]
        )

    @lru_cache()
    def _ipython_key_completions_(self) -> Set[str]:
        return set(elt["designator"] for elt in self.elements_list)
