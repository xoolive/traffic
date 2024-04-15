from __future__ import annotations

import logging
from typing import Dict

import geopandas as gpd

from shapely.geometry import shape
from shapely.ops import orient

from ...core.airspace import Airspace, Airspaces, ExtrudedPolygon
from . import ADDS_FAA_OpenData

_log = logging.getLogger(__name__)


class Airspace_Boundary(ADDS_FAA_OpenData, Airspaces):
    id_ = "67885972e4e940b2aa6d74024901c561_0"
    filename = "faa_airspace_boundary.json"

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

    def back(self) -> Dict[str, Airspace]:
        features = [elt for elt in self.json_contents()["features"]]
        airspaces: Dict[str, Airspace] = dict()

        for feat in features:
            name = feat["properties"]["NAME"].strip()

            airspace = Airspace(
                name=name,
                elements=[
                    ExtrudedPolygon(
                        orient(shape(feat["geometry"]), -1),
                        feat["properties"]["LOWER_VAL"]
                        if feat["properties"]["LOWER_VAL"] != -9998
                        else 0,
                        feat["properties"]["UPPER_VAL"]
                        if feat["properties"]["UPPER_VAL"] != -9998
                        else float("inf"),
                    )
                ],
                type_=feat["properties"]["TYPE_CODE"],
                designator=feat["properties"]["IDENT"],
                properties=feat["properties"],
            )

            if not airspace.shape.is_valid:
                _log.warning(f"Invalid shape part {name}, skipping...")
                continue

            if name in airspaces:
                airspaces[name] += airspace
            else:
                airspaces[name] = airspace

        return airspaces
