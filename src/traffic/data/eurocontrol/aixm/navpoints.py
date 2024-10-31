from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lxml import etree
from typing_extensions import Self

import pandas as pd

from ...basic.navaid import Navaids

_log = logging.getLogger(__name__)


class AIXMNavaidParser(Navaids):
    name: str = "aixm_navaids"
    filename: Path
    priority: int = 2
    _extensions: Optional[pd.DataFrame] = None

    @property
    def available(self) -> bool:
        if self.filename is None:
            return False

        dp_file = next(self.filename.glob("DesignatedPoint.BASELINE"), None)
        navaid_file = next(self.filename.glob("Navaid.BASELINE"), None)

        return dp_file is not None and navaid_file is not None

    @property
    def extensions(self) -> pd.DataFrame:
        if self._extensions is not None:
            return self._extensions

        cache_file = self.cache_dir / (self.filename.stem + "_navpoints.pkl")
        extension_file = self.cache_dir / (
            self.filename.stem + "_navpoints_extensions.pkl"
        )

        if not extension_file.exists():
            self.parse_data()
            if self._data is not None:
                self._data.to_pickle(cache_file)
            if self._extensions is not None:
                self._extensions.to_pickle(extension_file)
        else:
            _log.info("Loading aixm points database")

            self._extensions = pd.read_pickle(extension_file)

        return self._extensions

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        cache_file = self.cache_dir / (self.filename.stem + "_navpoints.pkl")
        extension_file = self.cache_dir / (
            self.filename.stem + "_navpoints_extensions.pkl"
        )

        if not cache_file.exists():
            self.parse_data()
            if self._data is not None:
                self._data.to_pickle(cache_file)
            if self._extensions is not None:
                self._extensions.to_pickle(extension_file)
        else:
            _log.info("Loading aixm points database")
            self._data = pd.read_pickle(cache_file)

        return self._data

    @classmethod
    def from_file(cls, filename: Union[Path, str], **kwargs: Any) -> Self:
        instance = cls(None)
        instance.filename = Path(filename)
        return instance

    def id_latlon(self, id_: str) -> None | tuple[float, float]:
        filtered = self.query(f"id == '{id_}'")
        if filtered is None:
            return None
        else:
            elt = filtered.data.iloc[0]
            return (elt.latitude, elt.longitude)

    def parse_data(self) -> None:
        dirname = Path(self.filename)
        all_points: Dict[str, Dict[str, Any]] = {}
        extensions: List[Dict[str, Any]] = []

        for filename in ["DesignatedPoint.BASELINE", "Navaid.BASELINE"]:
            if not (dirname / filename).exists():
                zippath = zipfile.ZipFile(
                    dirname.joinpath(f"{filename}.zip").as_posix()
                )
                zippath.extractall(dirname.as_posix())

        ns: Dict[str, str] = dict()

        # The versions for namespaces may be incremented and make everything
        # fail just for that reason!
        for _, (key, value) in etree.iterparse(
            (dirname / "DesignatedPoint.BASELINE").as_posix(),
            events=["start-ns"],
        ):
            ns[key] = value

        points = etree.parse((dirname / "DesignatedPoint.BASELINE").as_posix())

        for point in points.findall(
            "adrmsg:hasMember/aixm:DesignatedPoint", ns
        ):
            identifier = point.find("gml:identifier", ns)
            assert identifier is not None
            assert identifier.text is not None

            floats = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/"
                "aixm:location/aixm:Point/gml:pos",
                ns,
            )
            assert floats is not None
            assert floats.text is not None

            designator = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/aixm:designator",
                ns,
            )
            type_ = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/aixm:type",
                ns,
            )
            name_ = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/aixm:name",
                ns,
            )

            name = designator.text if designator is not None else None
            if name is None and name_ is not None:
                name = name_.text

            type_str = type_.text if type_ is not None else None

            coords = tuple(float(x) for x in floats.text.split())
            all_points[identifier.text] = {
                "latitude": coords[0],
                "longitude": coords[1],
                "name": name,
                "type": type_str,
                "id": identifier.text,
            }

            extension = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/"
                "aixm:extension",
                ns,
            )
            if extension is not None:
                for point_usage in extension.findall(
                    "adrext:DesignatedPointExtension/"
                    "adrext:pointUsage/adrext:PointUsage",
                    ns,
                ):
                    role = point_usage.find("adrext:role", ns)
                    elt = dict(id=identifier.text, role=role.text)
                    airspace = point_usage.find("adrext:reference_airspace", ns)
                    if airspace is not None:
                        airspace_ref = airspace.attrib["{%s}href" % ns["xlink"]]
                        elt["airspace"] = airspace_ref.split(":")[-1]
                    reference_border = point_usage.find(
                        "adrext:reference_border", ns
                    )
                    if reference_border is not None:
                        path = "adrext:AirspaceBorderCrossingObject/"
                        path += "adrext:{}edAirspace".format(
                            "enter" if role.text == "FRA_ENTRY" else "exit"
                        )
                        airspace = reference_border.find(path, ns)
                        assert airspace is not None
                        airspace_ref = airspace.attrib["{%s}href" % ns["xlink"]]
                        elt["airspace"] = airspace_ref.split(":")[-1]

                    extensions.append(elt)

        points = etree.parse((dirname / "Navaid.BASELINE").as_posix())

        for point in points.findall("adrmsg:hasMember/aixm:Navaid", ns):
            identifier = point.find("gml:identifier", ns)
            assert identifier is not None
            assert identifier.text is not None

            floats = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/"
                "aixm:location/aixm:ElevatedPoint/gml:pos",
                ns,
            )
            assert floats is not None
            assert floats.text is not None

            designator = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/aixm:designator", ns
            )
            type_ = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/aixm:type", ns
            )
            description = point.find(
                "aixm:timeSlice/aixm:NavaidTimeSlice/aixm:name", ns
            )

            name = designator.text if designator is not None else None
            type_str = type_.text if type_ is not None else None
            description_str = (
                description.text if description is not None else None
            )

            coords = tuple(float(x) for x in floats.text.split())
            all_points[identifier.text] = {
                "latitude": coords[0],
                "longitude": coords[1],
                "name": name,
                "type": type_str,
                "description": description_str,
                "id": identifier.text,
            }

            extension = point.find(
                "aixm:timeSlice/aixm:DesignatedPointTimeSlice/"
                "aixm:extension",
                ns,
            )
            if extension is not None:
                for point_usage in extension.findall(
                    "adrext:DesignatedPointExtension/"
                    "adrext:pointUsage/adrext:PointUsage",
                    ns,
                ):
                    role = point_usage.find("adrext:role", ns)
                    elt = dict(id=identifier.text, role=role.text)
                    airspace = point_usage.find("adrext:reference_airspace", ns)
                    if airspace is not None:
                        airspace_ref = airspace.attrib["{%s}href" % ns["xlink"]]
                        elt["airspace"] = airspace_ref.split(":")[-1]
                    reference_border = point_usage.find(
                        "adrext:reference_border", ns
                    )
                    if reference_border is not None:
                        path = "adrext:AirspaceBorderCrossingObject/"
                        path += "adrext:{}edAirspace".format(
                            "enter" if role.text == "FRA_ENTRY" else "exit"
                        )
                        airspace = reference_border.find(path, ns)
                        assert airspace is not None
                        airspace_ref = airspace.attrib["{%s}href" % ns["xlink"]]
                        elt["airspace"] = airspace_ref.split(":")[-1]

                        path += "adrext:{}edAirspace".format(
                            "exit" if role.text == "FRA_ENTRY" else "enter"
                        )
                        airspace = reference_border.find(path, ns)
                        assert airspace is not None
                        airspace_ref = airspace.attrib["{%s}href" % ns["xlink"]]
                        elt["other"] = airspace_ref.split(":")[-1]

                    extensions.append(elt)

        self._data = pd.DataFrame.from_records(
            point for point in all_points.values()
        )
        self._extensions = pd.DataFrame.from_records(extensions)

        return
