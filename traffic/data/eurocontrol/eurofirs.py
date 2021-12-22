import json
from pathlib import Path

from shapely.geometry import polygon, shape

from ...core.airspace import Airspace, ExtrudedPolygon

with Path(__file__).absolute().with_name("eurofirs.json").open("r") as fh:
    fir = json.load(fh)

eurofirs = {
    elt["properties"]["IDENT"]: Airspace(
        name=elt["properties"]["NAME"][:-4],  # Remove " FIR" at the end
        elements=[
            ExtrudedPolygon(
                polygon.orient(shape(elt["geometry"]), -1),
                int(elt["properties"]["LOWERLIMIT"]),
                int(elt["properties"]["UPPERLIMIT"]),
            )
        ],
        type_=elt["properties"]["TYPE"],
        designator=elt["properties"]["IDENT"],
        properties=elt["properties"],
    )
    for elt in fir["features"]
}
