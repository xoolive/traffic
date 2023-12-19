# ruff: noqa: E402
from __future__ import annotations

# %%
from pathlib import Path
from typing import Any

import geopandas as gpd
from traffic.data import aixm_airspaces

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform, unary_union

# %%

x = aixm_airspaces.data.loc[
    aixm_airspaces.data.designator == "BIRD",
    ["name", "type", "designator", "identifier"],
].iloc[0]

# Faroe issue in BIRD
aixm_airspaces.data.loc[
    aixm_airspaces.data.designator == "BIFAROER",
    ["name", "type", "designator", "identifier"],
] = x.values

# Jan Mayer issue in BIRD
aixm_airspaces.data.loc[
    aixm_airspaces.data.designator == "ENJA",
    ["name", "type", "designator", "identifier"],
] = x.values

aixm_airspaces.data.loc[
    aixm_airspaces.data["type"] == "UIR_P",
    ["name", "type", "designator"],
] = ["CANARIAS UIR", "UIR", "GCCC"]

for name in ["VCCF", "UTSD", "UTTR", "UHPP"]:
    aixm_airspaces.data.loc[
        aixm_airspaces.data.designator == name, "upper"
    ] = np.inf

# %%


def match180x(x: float) -> float:
    return 180 if x == 179.0 else (-180 if x == -179.0 else x)


def match90y(y: float) -> float:
    return 90 if y == 89.0 else (-90 if y == -89.0 else y)


def match180(x: Any, y: Any, z: None = None) -> tuple[Any, ...]:
    return tuple(
        filter(
            None,
            [
                tuple(match180x(x_) for x_ in x),
                tuple(match90y(y_) for y_ in y),
                z,
            ],
        )
    )


subset = aixm_airspaces.query(
    '(type == "FIR" or type == "UIR" or type == "NO_FIR" or '
    'designator == "BODO") and not (type == "FIR" and designator == "XXXX")'
)
assert subset is not None

gdf = (
    subset.consolidate()
    .assign(
        geometry=lambda gdf: gdf.geometry.apply(
            lambda geom: transform(match180, geom)
        )
    )
    .drop(columns=["identifier"])
    .groupby(["designator", "type", "upper", "lower", "name"], as_index=False)
    .agg(dict(geometry=unary_union))
)
gdf.loc[
    (gdf["type"] != "UIR") & (gdf["type"] != "FIR"),
    ["designator", "type", "name"],
] = (None, "NO_FIR", "")
gdf.loc[gdf["upper"] == np.nan, "upper"] = None
gdf.loc[gdf.designator == "BIRD", "geometry"] = unary_union(
    gdf.query('designator == "BIRD"').geometry
)
gdf = gdf.query("not(designator == 'BIRD' and (upper == 200 or lower == 200))")
# weird hole in LOVV
gdf.loc[gdf.designator == "LOVV", "geometry"] = Polygon(
    gdf.loc[gdf.designator == "LOVV", "geometry"].item().exterior
)
# bug in NZZO around South Pole
debug_nzzo = mapping(gdf.query('designator == "NZZO"').geometry.item())
gdf.loc[gdf.designator == "NZZO", ["geometry"]] = gpd.GeoSeries(
    [
        shape(
            {
                "type": "MultiPolygon",
                "coordinates": (
                    (debug_nzzo["coordinates"][0][0],),
                    (
                        (
                            (-157.0, -30.0),
                            (-131.0, -30.0),
                            (-131.0, -90.0),
                            (-180.0, -90.0),
                            (-180.0, -38.69555555555556),
                            (-180.0, -38.45),
                            (-180.0, -38.270833333333336),
                            (-180.0, -25.0),
                            (-175.67527777777778, -15.545833333333333),
                            (-171.0, -5.0),
                            (-157.0, -5.0),
                            (-157.0, -30.0),
                        ),
                    ),
                ),
            }
        )
    ]
).values


def format_name(line: pd.Series) -> str:
    name: str = line["name"]
    if name == "":
        return ""
    if name == "V W A REGION":
        return "Hanoi FIR"
    name = (
        name.replace("FLIGHT INFORMATION REGION", "FIR")
        .replace("FIR / UIR", "FIR/UIR")
        .replace("FT WORTH", "FORT WORTH")
    )
    if name.startswith("UIR "):
        name = name[4:] + " UIR"
    if name.startswith("FIR "):
        name = name[4:] + " FIR"
    if not name.endswith("FIR") and not name.endswith("UIR"):
        name = name + " " + line["type"]

    return name.title().replace("Fir", "FIR").replace("Uir", "UIR")


gdf = gdf.assign(name=[format_name(line) for _, line in gdf.iterrows()])

# %%
import topojson as tp

t = tp.Topology(gdf)
Path("worldfirs.json").write_text(t.to_json().replace("Infinity", "null"))
