from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from ...core.airspace import Airspaces


class Eurofirs(Airspaces):
    def __init__(self, data: gpd.GeoDataFrame | None = None) -> None:
        self.data = data
        if data is None:
            current_file = Path(__file__).absolute()
            with current_file.with_name("eurofirs.json").open("r") as fh:
                self.fir_json = json.load(fh)
            self.data = (
                gpd.GeoDataFrame.from_features(self.fir_json)
                .rename(
                    columns=dict(
                        NAME="name",
                        LOWERLIMIT="lower",
                        UPPERLIMIT="upper",
                        TYPE="type",
                        IDENT="designator",
                    )
                )
                .assign(
                    latitude=lambda df: df.geometry.centroid.y,
                    longitude=lambda df: df.geometry.centroid.x,
                    lower=lambda df: df.lower.astype(float),
                    upper=lambda df: df.upper.astype(float).replace(
                        999, float("inf")
                    ),
                )
                .drop(columns=["UPPERUNIT", "LOWERUNIT", "EFFECTDATE", "ICAO"])
                .set_geometry("geometry")
            )


eurofirs = Eurofirs()
