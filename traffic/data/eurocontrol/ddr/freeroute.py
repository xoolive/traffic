import re
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Set, Tuple

import geopandas as gpd

import pandas as pd

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

        frp_file = next(self.nm_path.glob("Free_Route_*.frp"), None)
        if frp_file is None:
            raise RuntimeError(
                f"No Free_Route_*.frp file found in {self.nm_path}"
            )
        self.read_frp(frp_file)

        self.initialized = True

        self.fra = gpd.GeoDataFrame.from_records(
            [
                {"FRA": k, "geometry": self[k].shape}  # type: ignore
                for k in self.elements.keys()
            ]
        )

    def read_frp(self, filename: Path) -> None:

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
        coords = (
            tab.query("latitude.notnull()")[["latitude", "longitude"]]
            .sum(axis=1)
            .apply(parse_coordinates)
        )

        self.frp = pd.concat(
            [
                tab.query("latitude.notnull()").assign(
                    latitude=coords.str[0], longitude=coords.str[1]
                ),
                tab.query("latitude.isnull()")
                .drop(columns=["latitude", "longitude"])
                .merge(
                    nm_navaids.data.drop(columns=["type", "description"]),
                    on="name",
                ),
                pd.concat(
                    [
                        df_.query('type in ["AD", "A", "D"]').iloc[:, :3],
                        a.rename("airport"),
                    ],
                    axis=1,
                )
                .explode("airport")
                .merge(
                    nm_navaids.data.drop(columns=["type", "description"]),
                    on="name",
                ),
            ]
        )

    def __getattr__(self, attr: str) -> Any:
        if attr in ["fra", "frp"]:
            self.init_cache()
            return getattr(self, attr)
        raise AttributeError(attr)

    @lru_cache()
    def _ipython_key_completions_(self) -> Set[str]:
        return {*self.elements.keys()}
