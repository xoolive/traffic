from __future__ import annotations

import logging
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from lxml import etree
from typing_extensions import Self

import pandas as pd

from ....core.structure import Navaid, Route
from ... import aixm_navaids
from ...basic.airways import Airways

_log = logging.getLogger(__name__)


class AIXMRoutesParser(Airways):
    name: str = "aixm_airways"
    filename: Path
    priority: int = 2
    cache_dir: Path

    @classmethod
    def from_file(cls, filename: str | Path, **kwargs: Any) -> Self:
        instance = cls(None)
        instance.filename = Path(filename)
        return instance

    @lru_cache()
    def __getitem__(self, name: str) -> None | Route:
        output = self.data.query("name == @name")
        if output.shape[0] == 0:
            return None

        cumul = []
        for x, d in output.groupby("routeFormed"):
            keys = dict(zip(d.start_id, d.end_id))
            start_set = set(d.start_id) - set(d.end_id)
            start = start_set.pop()
            unfold_list = [{"id": start}]
            while start is not None:
                start = keys.get(start, None)
                if start is not None:
                    unfold_list.append({"id": start})
            cumul.append(
                pd.DataFrame.from_records(unfold_list)
                .merge(aixm_navaids.data)
                .assign(routeFormed=x, route=d.name.max())
            )
        output = pd.concat(cumul)

        return Route(
            name,
            list(
                Navaid(
                    x["name"],
                    x["type"],
                    x["latitude"],
                    x["longitude"],
                    0,
                    None,
                    None,
                    None,
                )
                for _, x in output.iterrows()
            ),
        )

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        cache_file = self.cache_dir / (self.filename.stem + "_airways.parquet")
        if not cache_file.exists():
            self._data = self.parse_data()
            if self._data is not None:
                self._data.to_parquet(cache_file)
        else:
            _log.info("Loading aixm route database")
            self._data = pd.read_parquet(cache_file)

        return self._data

    def parse_data(self) -> pd.DataFrame:
        assert self.filename is not None
        if self.filename is None or not self.filename.exists():
            msg = "Edit configuration file with AIXM directory"
            raise RuntimeError(msg)

        route_definition = "Route.BASELINE"
        segment_definition = "RouteSegment.BASELINE"

        if not (self.filename / route_definition).exists():
            zippath = zipfile.ZipFile(
                self.filename.joinpath(f"{route_definition}.zip").as_posix()
            )
            zippath.extractall(self.filename.as_posix())

        # The versions for namespaces may be incremented and make everything
        # fail just for that reason!
        ns: dict[str, str] = {}
        for _, (key, value) in etree.iterparse(
            (self.filename / route_definition).as_posix(),
            events=["start-ns"],
        ):
            ns[key] = value

        tree = etree.parse((self.filename / route_definition).as_posix())
        self.data_routes = pd.DataFrame.from_records(
            self.parse_routes(tree, ns)
        )

        if not (self.filename / segment_definition).exists():
            zippath = zipfile.ZipFile(
                self.filename.joinpath(f"{segment_definition}.zip").as_posix()
            )
            zippath.extractall(self.filename.as_posix())

        ns.clear()

        for _, (key, value) in etree.iterparse(
            (self.filename / segment_definition).as_posix(),
            events=["start-ns"],
        ):
            ns[key] = value

        tree = etree.parse((self.filename / segment_definition).as_posix())
        self.data_segments = pd.DataFrame.from_records(
            self.parse_segments(tree, ns)
        )

        start = pd.DataFrame(
            {
                "identifier": self.data_segments["identifier"],
                "start_pt": self.data_segments["start_designatedPoint"],
                "start_nav": self.data_segments["start_navaid"],
            }
        )
        start_merged = start["start_pt"].fillna(start["start_nav"])
        self.data_segments["start_id"] = start_merged
        self.data_segments = self.data_segments.drop(
            columns=["start_designatedPoint", "start_navaid"]
        )

        end = pd.DataFrame(
            {
                "identifier": self.data_segments["identifier"],
                "end_pt": self.data_segments["end_designatedPoint"],
                "end_nav": self.data_segments["end_navaid"],
            }
        )
        end_merged = end["end_pt"].fillna(end["end_nav"])
        self.data_segments["end_id"] = end_merged
        self.data_segments = self.data_segments.drop(
            columns=["end_designatedPoint", "end_navaid"]
        )

        to_merge_start = pd.DataFrame(
            {
                "start_id": aixm_navaids.data["id"],
                "start_name": aixm_navaids.data["name"],
            }
        )
        to_merge_end = pd.DataFrame(
            {
                "end_id": aixm_navaids.data["id"],
                "end_name": aixm_navaids.data["name"],
            }
        )

        self.data_segments = self.data_segments.merge(
            to_merge_start, on="start_id"
        )
        self.data_segments = self.data_segments.merge(to_merge_end, on="end_id")

        return (
            pd.concat(
                [
                    self.data_routes.query("prefix.notnull()").eval(
                        "name = prefix + secondLetter + number"
                    ),
                    self.data_routes.query("prefix.isnull()").eval(
                        "name =  secondLetter + number"
                    ),
                ]
            )
            .drop(columns=["prefix", "secondLetter", "number"])
            .rename(columns={"identifier": "routeFormed"})
            .merge(self.data_segments, on="routeFormed")
        )

    def parse_routes(
        self, tree: etree.ElementTree, ns: dict[str, str]
    ) -> Iterator[dict[str, Any]]:
        for point in tree.findall("adrmsg:hasMember/aixm:Route", ns):
            identifier = point.find("gml:identifier", ns)
            assert identifier is not None
            assert identifier.text is not None

            designatorPrefix = point.find(
                "aixm:timeSlice/aixm:RouteTimeSlice/aixm:designatorPrefix",
                ns,
            )

            designatorSecondLetter = point.find(
                "aixm:timeSlice/aixm:RouteTimeSlice/aixm:designatorSecondLetter",
                ns,
            )

            designatorNumber = point.find(
                "aixm:timeSlice/aixm:RouteTimeSlice/aixm:designatorNumber", ns
            )

            multipleIdentifier = point.find(
                "aixm:timeSlice/aixm:RouteTimeSlice/aixm:multipleIdentifier", ns
            )

            beginPosition = point.find(
                "aixm:timeSlice/aixm:RouteTimeSlice/gml:validTime/gml:TimePeriod/gml:beginPosition",
                ns,
            )

            endPosition = point.find(
                "aixm:timeSlice/aixm:RouteTimeSlice/gml:validTime/gml:TimePeriod/gml:endPosition",
                ns,
            )

            designatorPrefix_str = (
                designatorPrefix.text if designatorPrefix is not None else None
            )
            designatorSecondLetter_str = (
                designatorSecondLetter.text
                if designatorSecondLetter is not None
                else None
            )
            designatorNumber_str = (
                designatorNumber.text if designatorNumber is not None else None
            )
            multipleIdentifier_str = (
                multipleIdentifier.text
                if multipleIdentifier is not None
                else None
            )
            beginPosition_str = (
                beginPosition.text if beginPosition is not None else None
            )

            endPosition_str = (
                endPosition.text if endPosition is not None else None
            )

            yield {
                "identifier": identifier.text,
                "prefix": designatorPrefix_str,
                "secondLetter": designatorSecondLetter_str,
                "number": designatorNumber_str,
                "multipleIdentifier": multipleIdentifier_str,
                "beginPosition": beginPosition_str,
                "endPosition": endPosition_str,
            }

    def parse_segments(
        self, tree: etree.ElementTree, ns: dict[str, str]
    ) -> Iterator[dict[str, Any]]:
        for point in tree.findall("adrmsg:hasMember/aixm:RouteSegment", ns):
            identifier = point.find("gml:identifier", ns)
            assert identifier is not None
            assert identifier.text is not None

            beginPosition = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/gml:validTime/gml:TimePeriod/gml:beginPosition",
                ns,
            )

            endPosition = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/gml:validTime/"
                "gml:TimePeriod/gml:endPosition",
                ns,
            )

            upperLimit = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:upperLimit",
                ns,
            )

            lowerLimit = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:lowerLimit",
                ns,
            )

            routeFormed = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:routeFormed",
                ns,
            )

            start_designatedPoint = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:start/"
                "aixm:EnRouteSegmentPoint/aixm:pointChoice_fixDesignatedPoint",
                ns,
            )

            end_designatedPoint = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:end/"
                "aixm:EnRouteSegmentPoint/aixm:pointChoice_fixDesignatedPoint",
                ns,
            )

            start_navaid = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:start/"
                "aixm:EnRouteSegmentPoint/aixm:pointChoice_navaidSystem",
                ns,
            )

            end_navaid = point.find(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:end/"
                "aixm:EnRouteSegmentPoint/aixm:pointChoice_navaidSystem",
                ns,
            )

            directions = point.findall(
                "aixm:timeSlice/aixm:RouteSegmentTimeSlice/aixm:availability/"
                "aixm:RouteAvailability/aixm:direction",
                ns,
            )

            if directions is None or len(directions) < 1:
                direction_str = "BOTH"
            elif directions is not None and len(directions) > 1:
                direction_str = directions[0].text
                for d in directions:
                    if d.text != direction_str:
                        direction_str = "BOTH"
            else:
                direction_str = directions[0].text

            # direction_str = direction.text if direction is not None else None

            beginPosition_str = (
                beginPosition.text if beginPosition is not None else None
            )

            endPosition_str = (
                endPosition.text if endPosition is not None else None
            )

            upperLimit_str = upperLimit.text if upperLimit is not None else None

            lowerLimit_str = lowerLimit.text if lowerLimit is not None else None

            routeFormed_str = (
                routeFormed.get("{http://www.w3.org/1999/xlink}href").split(
                    ":"
                )[2]
                if routeFormed is not None
                else None
            )

            start_designatedPoint_str = (
                start_designatedPoint.get(
                    "{http://www.w3.org/1999/xlink}href"
                ).split(":")[2]
                if start_designatedPoint is not None
                else None
            )

            start_navaid_str = (
                start_navaid.get("{http://www.w3.org/1999/xlink}href").split(
                    ":"
                )[2]
                if start_navaid is not None
                else None
            )

            end_designatedPoint_str = (
                end_designatedPoint.get(
                    "{http://www.w3.org/1999/xlink}href"
                ).split(":")[2]
                if end_designatedPoint is not None
                else None
            )

            end_navaid_str = (
                end_navaid.get("{http://www.w3.org/1999/xlink}href").split(":")[
                    2
                ]
                if end_navaid is not None
                else None
            )

            yield {
                "identifier": identifier.text,
                "beginPosition": beginPosition_str,
                "endPosition": endPosition_str,
                "upperLimit": upperLimit_str,
                "lowerLimit": lowerLimit_str,
                "start_designatedPoint": start_designatedPoint_str,
                "start_navaid": start_navaid_str,
                "end_designatedPoint": end_designatedPoint_str,
                "end_navaid": end_navaid_str,
                "routeFormed": routeFormed_str,
                "direction": direction_str,
            }
