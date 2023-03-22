from __future__ import annotations

import logging

import geopandas as gpd

from ...core.airspace import Airspaces
from . import ADDS_FAA_OpenData

_log = logging.getLogger(__name__)


class Class_Airspaces(ADDS_FAA_OpenData, Airspaces):
    id_ = "c6a62360338e408cb1512366ad61559e_0"
    filename = "faa_class_airspace.json"

    def __init__(self, data: gpd.GeoDataFrame | None = None) -> None:
        super().__init__()
        self.data = data
        if data is None:
            self.data = (
                gpd.GeoDataFrame.from_features(self.json_contents())
                .rename(
                    columns=dict(
                        NAME="name",
                        LOWER_VAL="lower",
                        UPPER_VAL="upper",
                        TYPE_CODE="type",
                        IDENT="designator",
                    )
                )
                .assign(
                    latitude=lambda df: df.geometry.centroid.y,
                    longitude=lambda df: df.geometry.centroid.x,
                    name=lambda df: df.name.str.strip(),
                    lower=lambda df: df.lower.replace(-9998, 0),
                    upper=lambda df: df.upper.replace(-9998, float("inf")),
                )
                .set_geometry("geometry")
            )
