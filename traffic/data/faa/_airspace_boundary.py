import logging
from typing import Dict

from shapely.geometry import polygon, shape

from ...core.airspace import Airspace, ExtrudedPolygon
from . import ADDS_FAA_OpenData


class Airspace_Boundary(ADDS_FAA_OpenData):

    id_ = "67885972e4e940b2aa6d74024901c561_0"
    filename = "faa_airspace_boundary.json"

    def get_data(self) -> Dict[str, Airspace]:

        features = [elt for elt in self.json_contents()["features"]]
        airspaces: Dict[str, Airspace] = dict()

        for feat in features:
            name = feat["properties"]["NAME"].strip()

            airspace = Airspace(
                name=name,
                elements=[
                    ExtrudedPolygon(
                        polygon.orient(shape(feat["geometry"]), -1),
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
                logging.warning(f"Invalid shape part {name}, skipping...")
                continue

            if name in airspaces:
                airspaces[name] += airspace
            else:
                airspaces[name] = airspace

        return airspaces
