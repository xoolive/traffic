from __future__ import annotations

import logging
from typing import Dict

import geopandas as gpd

from shapely.geometry import shape
from shapely.ops import orient

from ...core.airspace import Airspace, Airspaces, ExtrudedPolygon
from . import ADDS_FAA_OpenData

_log = logging.getLogger(__name__)


class FAA_Airspace(ADDS_FAA_OpenData, Airspaces):
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


class Airspace_Boundary(FAA_Airspace):
    id_ = "67885972e4e940b2aa6d74024901c561_0"
    filename = "faa_airspace_boundary.json"


class Class_Airspace(FAA_Airspace):
    id_ = "c6a62360338e408cb1512366ad61559e_0"
    filename = "faa_class_airspace.json"


class Special_Use_Airspace(FAA_Airspace):
    id_ = "dd0d1b726e504137ab3c41b21835d05b_0"
    filename = "faa_special_use_airspace.json"


class Route_Airspace(FAA_Airspace):
    id_ = "8bf861bb9b414f4ea9f0ff2ca0f1a851_0"
    filename = "faa_route_airspace.json"


class Prohibited_Airspace(FAA_Airspace):
    id_ = "354ee0c77484461198ebf728a2fca50c_0"
    filename = "faa_prohibited_airspace.json"
