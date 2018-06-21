import json
from pathlib import Path

from shapely.geometry import shape

from ...core.airspace import ExtrudedPolygon, Airspace

with Path(__file__).absolute().with_name("firs.json").open("r") as fh:
    fir = json.loads("".join(fh.readlines()))

eurofirs = {
    elt["properties"]["IDENT"]: Airspace(
        name=elt["properties"]["NAME"][:-4],  # Remove " FIR" at the end
        elements=[
            ExtrudedPolygon(
                shape(elt["geometry"]),
                int(elt["properties"]["LOWERLIMIT"]),
                int(elt["properties"]["UPPERLIMIT"]),
            )
        ],
        type_=elt["properties"]["TYPE"],
    )
    for elt in fir["features"]
}
