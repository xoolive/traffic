import json
from pathlib import Path

from shapely.geometry import shape
from .core import ExtrudedPolygon, Sector


with Path(__file__).absolute().with_name("firs.json").open('r') as fh:
    fir = json.loads("".join(fh.readlines()))

eurofirs = {
    elt["properties"]["IDENT"]: Sector(
        name=elt["properties"]["NAME"][:-4],  # Remove " FIR" at the end
        elements=[
            ExtrudedPolygon(
                shape(elt["geometry"]),
                elt["properties"]["LOWERLIMIT"],
                elt["properties"]["UPPERLIMIT"],
            )
        ],
        type_=elt["properties"]["TYPE"],
    )
    for elt in fir["features"]
}
