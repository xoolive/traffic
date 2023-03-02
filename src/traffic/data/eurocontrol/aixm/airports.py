from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
from lxml import etree

import pandas as pd

from ....core.mixins import DataFrameMixin


class AIXMAirportParser(DataFrameMixin):
    cache_dir: Path

    def __init__(
        self,
        data: None | pd.DataFrame,
        aixm_path: None | Path,
    ) -> None:
        self.data = data
        self.aixm_path = aixm_path

        if data is None:
            if aixm_path is None or not aixm_path.exists():
                msg = "Edit configuration file with AIXM directory"
                raise RuntimeError(msg)

            # Read file in cache if present
            airports_file = self.cache_dir / f"{aixm_path.stem}_airports.pkl"
            if airports_file.exists():
                self.data = gpd.GeoDataFrame(pd.read_pickle(airports_file))
                return

            airport_definition = "AirportHeliport.BASELINE"
            if not (aixm_path / airport_definition).exists():
                zippath = zipfile.ZipFile(
                    aixm_path.joinpath(f"{airport_definition}.zip").as_posix()
                )
                zippath.extractall(aixm_path.as_posix())

            # The versions for namespaces may be incremented and make everything
            # fail just for that reason!
            ns: dict[str, str] = {}
            for _, (key, value) in etree.iterparse(
                (aixm_path / airport_definition).as_posix(),
                events=["start-ns"],
            ):
                ns[key] = value

            tree = etree.parse((aixm_path / airport_definition).as_posix())

            self.data = pd.DataFrame.from_records(self.parse_tree(tree, ns))
            self.data.to_pickle(airports_file)

    def parse_tree(
        self, tree: etree.ElementTree, ns: dict[str, str]
    ) -> Iterator[dict[str, Any]]:
        for elt in tree.findall("adrmsg:hasMember/aixm:AirportHeliport", ns):
            identifier = elt.find("gml:identifier", ns)
            assert identifier is not None
            assert identifier.text is not None

            apt = elt.find("aixm:timeSlice/aixm:AirportHeliportTimeSlice", ns)
            if apt is None:
                continue

            nameElt = apt.find("aixm:name", ns)
            icaoElt = apt.find("aixm:locationIndicatorICAO", ns)
            iataElt = apt.find("aixm:designatorIATA", ns)
            typeElt = apt.find("aixm:controlType", ns)
            cityElt = apt.find("aixm:servedCity/aixm:City/aixm:name", ns)
            posElt = apt.find("aixm:ARP/aixm:ElevatedPoint/gml:pos", ns)
            altElt = apt.find("aixm:ARP/aixm:ElevatedPoint/aixm:elevation", ns)

            if (
                posElt is None
                or posElt.text is None
                or altElt is None
                or altElt.text is None
                or icaoElt is None
            ):
                continue

            coords = tuple(float(x) for x in posElt.text.split())

            yield dict(
                identifier=identifier.text,
                latitude=coords[0],
                longitude=coords[1],
                altitude=float(altElt.text),
                iata=iataElt.text if iataElt is not None else None,
                icao=icaoElt.text,
                name=nameElt.text if nameElt is not None else None,
                city=cityElt.text if cityElt is not None else None,
                type=typeElt.text if typeElt is not None else None,
            )
