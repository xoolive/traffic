from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd

from shapely.geometry import polygon, shape

from ...core.airspace import Airspace, ExtrudedPolygon


class Eurofirs:
    def __init__(self) -> None:
        current_file = Path(__file__).absolute()
        with current_file.with_name("eurofirs.json").open("r") as fh:
            self.fir_json = json.load(fh)

    def _repr_html_(self) -> Any:
        return self.data._repr_html_()

    def __iter__(self) -> Iterator[Airspace]:
        for elt in self.fir_json["features"]:
            yield Airspace(
                name=elt["properties"]["NAME"],
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

    def __getitem__(self, name: str) -> None | Airspace:
        return next((elt for elt in self if elt.designator == name), None)

    @property
    def data(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame.from_features(self.fir_json).assign(
            latitude=lambda df: df.geometry.centroid.y,
            longitude=lambda df: df.geometry.centroid.x,
        )

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        return self.data.__geo_interface__  # type: ignore


eurofirs = Eurofirs()
