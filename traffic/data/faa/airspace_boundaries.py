import json
import logging
from typing import Dict

import requests
from shapely.geometry import shape

from ...core.airspace import Airspace, ExtrudedPolygon
from .. import cache_dir

id_ = "67885972e4e940b2aa6d74024901c561_0"

website = f"https://ais-faa.opendata.arcgis.com/datasets/{id_}"
json_url = f"https://opendata.arcgis.com/datasets/{id_}.geojson"


def get_airspaces() -> Dict[str, Airspace]:

    filename = cache_dir / "faa_airspaces.json"
    if filename.exists():
        with filename.open("r") as fh:
            json_contents = json.load(fh)
    else:
        logging.warning(
            f"Downloading data from {website}. Please check terms of use."
        )
        c = requests.get(json_url)
        c.raise_for_status()
        json_contents = c.json()
        with filename.open("w") as fh:
            json.dump(json_contents, fh)

    features = [elt for elt in json_contents["features"]]
    airspaces: Dict[str, Airspace] = dict()

    for feat in features:
        name = feat["properties"]["NAME"].strip()

        airspace = Airspace(
            name=name,
            elements=[
                ExtrudedPolygon(
                    shape(feat["geometry"]),
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

        if name in airspaces:
            airspaces[name] += airspace
        else:
            airspaces[name] = airspace

    return airspaces
