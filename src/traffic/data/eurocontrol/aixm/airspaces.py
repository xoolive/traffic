from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, List, Tuple, TypeVar

import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
from lxml import etree

import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import orient, unary_union

from ....core import tqdm
from ....core.airspace import (
    Airspace,
    Airspaces,
    ExtrudedPolygon,
    unary_union_with_alt,
)
from ... import aixm_navaids

T = TypeVar("T", bound="AIXMAirspaceParser")


def get_coordinates(lr: Any, ns: dict[str, str]) -> Polygon:
    coords: List[Tuple[float, ...]] = []
    gml, xlink = ns["gml"], ns["xlink"]
    for point in lr.iter():
        if point.tag in ("{%s}pos" % (gml), "{%s}pointProperty" % (gml)):
            if point.tag.endswith("pos"):
                coords.append(tuple(float(x) for x in point.text.split()))
            else:
                points = point.attrib["{%s}href" % (xlink)]

                latlon = aixm_navaids.id_latlon(points.split(":")[2])
                assert latlon is not None
                coords.append(latlon)

    return orient(Polygon([(lon, lat) for lat, lon in coords]), -1)


class AIXMAirspaceParser(Airspaces):
    cache_dir: Path

    def __init__(
        self, data: None | GeoDataFrame, aixm_path: None | Path = None
    ) -> None:
        self.data = data

        if data is None:
            if aixm_path is None or not aixm_path.exists():
                msg = "Edit configuration file with AIXM directory"
                raise RuntimeError(msg)

            # Read file in cache if present
            airspace_file = self.cache_dir / f"{aixm_path.stem}_airspaces.pkl"
            if airspace_file.exists():
                self.data = gpd.GeoDataFrame(pd.read_pickle(airspace_file))
                return

            airspace_definition = "Airspace.BASELINE"
            if not (aixm_path / airspace_definition).exists():
                zippath = zipfile.ZipFile(
                    aixm_path.joinpath(f"{airspace_definition}.zip").as_posix()
                )
                zippath.extractall(aixm_path.as_posix())

            # The versions for namespaces may be incremented and make everything
            # fail just for that reason!
            ns: dict[str, str] = {}
            for _, (key, value) in etree.iterparse(
                (aixm_path / airspace_definition).as_posix(),
                events=["start-ns"],
            ):
                ns[key] = value

            tree = etree.parse((aixm_path / airspace_definition).as_posix())

            self.data = gpd.GeoDataFrame.from_records(self.parse_tree(tree, ns))
            self.data.to_pickle(airspace_file)

    def __getitem__(self, name: str) -> None | Airspace:
        # in this case, running consolidate() on the whole dataset is not
        # reasonable, but it still works if we run it after the query
        subset = self.query(f'designator == "{name}"')
        if subset is None:
            return None

        return Airspace(
            elements=unary_union_with_alt(
                [
                    ExtrudedPolygon(line.geometry, line.lower, line.upper)
                    for _, line in subset.consolidate().data.iterrows()
                ]
            ),
            name=subset.data["name"].max(),
            type_=subset.data["type"].max(),
            designator=subset.data["designator"].max(),
        )

    def consolidate(self: T) -> T:
        if self.data.geometry.notnull().any():
            return self

        # Beware of circular import
        from ... import aixm_airspaces

        def consolidate_rec(data: pd.DataFrame) -> pd.DataFrame:
            # When no geometry information is provided, it may be provided by
            # referenced components. The cascading is done here, through a
            # recursive function.

            merged_data = data.merge(
                aixm_airspaces.data,
                left_on="component",
                right_on="identifier",
                suffixes=["", "_2"],
            ).drop(columns=["designator_2", "identifier_2"])

            null_geom_idx = merged_data.geometry.isnull()
            merged_data.loc[null_geom_idx, "geometry"] = merged_data.loc[
                null_geom_idx, "geometry_2"
            ]
            merged_data.loc[null_geom_idx, "geometry_2"] = None

            # type is not PART
            null_bounds_idx = merged_data.upper.isnull()
            merged_data.loc[null_bounds_idx, "upper"] = merged_data.loc[
                null_bounds_idx, "upper_2"
            ]
            merged_data.loc[null_bounds_idx, "upper_2"] = None
            merged_data.loc[null_bounds_idx, "lower"] = merged_data.loc[
                null_bounds_idx, "lower_2"
            ]
            merged_data.loc[null_bounds_idx, "lower_2"] = None

            null_geom_idx = merged_data.geometry.isnull()
            if null_geom_idx.sum() > 0:
                sub = (
                    merged_data.loc[null_geom_idx, :]
                    .drop(
                        columns=[
                            "upper_2",
                            "lower_2",
                            "type_2",
                            "name_2",
                            "geometry_2",
                            "component",
                        ]
                    )
                    .rename(columns=dict(component_2="component"))
                )
                cmp_sub = consolidate_rec(sub)

                for _, x in cmp_sub.iterrows():
                    merged_data.loc[
                        merged_data.component_2 == x.component, "geometry"
                    ] = x.geometry

            return merged_data.dropna(axis=1, how="all")

        new_data = consolidate_rec(pd.DataFrame(self.data))
        columns = ["designator", "upper", "lower", "type", "identifier"]

        name_table = None
        if "name" in new_data.columns:
            name_table = new_data[["identifier", "name"]].drop_duplicates()

        result = GeoDataFrame(
            new_data.groupby(columns)
            .agg(dict(geometry=unary_union))
            .reset_index()
        )
        if name_table is not None:
            result = result.merge(name_table)

        return self.__class__(result)

    def parse_tree(
        self, tree: etree.ElementTree, ns: dict[str, str]
    ) -> Iterable[dict[str, Any]]:
        for airspace in tqdm(
            tree.findall("adrmsg:hasMember/aixm:Airspace", ns),
            desc="Parsing definition file",
        ):
            identifier = airspace.find("gml:identifier", ns)
            for ts in airspace.findall(
                "aixm:timeSlice/aixm:AirspaceTimeSlice", ns
            ):
                designator_ = ts.find("aixm:designator", ns)
                type_ = ts.find("aixm:type", ns)
                name_ = ts.find("aixm:name", ns)

                for block in ts.findall(
                    "aixm:geometryComponent/aixm:AirspaceGeometryComponent/"
                    "aixm:theAirspaceVolume/aixm:AirspaceVolume",
                    ns,
                ):
                    upper_elt = block.find("aixm:upperLimit", ns)
                    lower_elt = block.find("aixm:lowerLimit", ns)

                    upper = (
                        (
                            float(upper_elt.text)
                            if re.match(r"\d+", upper_elt.text)
                            else float("inf")
                        )
                        if upper_elt is not None and upper_elt.text is not None
                        else None
                    )

                    lower = (
                        (
                            float(lower_elt.text)
                            if re.match(r"\d+", lower_elt.text)
                            else 0  # float("-inf")
                        )
                        if lower_elt is not None and lower_elt.text is not None
                        else None
                    )

                    # This should be a part?
                    for lr in block.findall(  # sub -> block
                        "aixm:horizontalProjection/aixm:Surface/"
                        "gml:patches/gml:PolygonPatch/gml:exterior/"
                        "gml:LinearRing",
                        ns,
                    ):
                        polygon = get_coordinates(lr, ns)

                        yield {
                            "name": name_.text if name_ is not None else None,
                            "type": type_.text if type_ is not None else None,
                            "designator": designator_.text
                            if designator_ is not None
                            else None,
                            "identifier": identifier.text,
                            "geometry": polygon,
                        }

                    for component in block.findall(
                        "aixm:contributorAirspace/"
                        "aixm:AirspaceVolumeDependency/"
                        "aixm:theAirspace",
                        ns,
                    ):
                        key = component.attrib[
                            "{http://www.w3.org/1999/xlink}href"
                        ]
                        key = key.split(":")[2]
                        yield {
                            "name": name_.text if name_ is not None else None,
                            "type": type_.text if type_ is not None else None,
                            "designator": designator_.text
                            if designator_ is not None
                            else None,
                            "upper": upper,
                            "lower": lower,
                            "identifier": identifier.text,
                            "component": key,
                        }
