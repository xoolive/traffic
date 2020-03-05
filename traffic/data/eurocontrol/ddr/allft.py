import re
from io import BytesIO
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
from tqdm.autonotebook import tqdm

from ....core import Flight
from ....core.mixins import DataFrameMixin, _HBox
from .so6 import _prepare_libarchive

AllFTTypeVar = TypeVar("AllFTTypeVar", bound="AllFT")


allft_fields = list(
    x
    for x in (Path(__file__).parent / "allft_fields.txt")
    .read_text()
    .split("\n")
    if x != ""
)


def parse_date(x):
    return pd.to_datetime(x).dt.tz_localize("utc")


def parse_coordinates(elt: str) -> Tuple[float, float]:
    pattern = r"(\d{6})([N,S])(\d{7})([E,W])$"
    x = re.match(pattern, elt)
    assert x is not None
    lat_, lat_sign = x.group(1), 1 if x.group(2) == "N" else -1
    lon_, lon_sign = x.group(3), 1 if x.group(4) == "E" else -1

    lat = lat_sign * (
        int(lat_[:2]) + int(lat_[2:4]) / 60 + int(lat_[4:]) / 3600
    )
    lon = lon_sign * (
        int(lon_[:3]) + int(lon_[3:5]) / 60 + int(lon_[5:]) / 3600
    )

    return lat, lon


class FlightInfo(DataFrameMixin):
    def __init__(self, data: pd.Series):
        self.data = data

    @property
    def origin(self) -> str:
        return self.data.origin

    @property
    def destination(self) -> str:
        return self.data.destination

    @property
    def callsign(self) -> str:
        return self.data.callsign

    @property
    def icao24(self) -> Optional[str]:
        if self.data.icao24 == self.data.icao24:
            return self.data.icao24
        else:
            return None

    @property
    def ifpsId(self) -> str:
        return self.data.ifpsId

    @property
    def flight_id(self) -> str:
        return self.data.ifpsId

    def _repr_html_(self):
        from ....data import aircraft, airports

        title = f"<h4><b>Flight {self.flight_id}</b> "
        title += f"({self.origin} → "
        title += f"{self.destination})</h4>"
        title += f"callsign: {self.callsign}<br/>"
        title += f" from {airports[self.origin]}<br/>"
        title += f" to {airports[self.destination]}<br/><br/>"

        cumul = list()
        if self.icao24 is not None:
            cumul.append(aircraft[self.icao24].T)

        cumul.append(
            pd.DataFrame(
                self.data[
                    [
                        "ifpsId",
                        "AOBT",
                        "IOBT",
                        "COBT",
                        "EOBT",
                        "flightState",
                        "mostPenalizingRegulationId",
                    ]
                ]
            )
        )

        return title + _HBox(*cumul)._repr_html_()

    def allFtPointProfile(self, name: str) -> Flight:
        if name not in ["ftfm", "rtfm", "ctfm"]:
            raise ValueError(f"{name} must be one of ftfm, rtfm and ctfm.")
        return Flight(
            pd.DataFrame.from_records(
                [
                    x.split(":")
                    for x in self.data[name + "AllFtPointProfile"].split()
                ],
                columns=[
                    "timestamp",
                    "point",
                    "route",
                    "flightLevel",
                    "pointDistance",
                    "pointType",
                    "geoPointId",
                    "relDist",
                    "isVisible",
                ],
                coerce_float=True,
            )
            .assign(
                timestamp=lambda df: parse_date(df.timestamp),
                flightLevel=lambda df: df.flightLevel.astype(int),
                pointDistance=lambda df: df.pointDistance.astype(int),
                altitude=lambda df: df.flightLevel.astype(int) * 100,
                geoPointId=lambda df: df.geoPointId.apply(parse_coordinates),
                latitude=lambda df: df.geoPointId.apply(itemgetter(0)),
                longitude=lambda df: df.geoPointId.apply(itemgetter(1)),
                icao24=self.icao24,
                callsign=self.callsign,
                flight_id=self.ifpsId,
                origin=self.origin,
                destination=self.destination,
            )
            .drop(columns=["geoPointId", "relDist", "isVisible"])
        )


class AllFT(DataFrameMixin):
    __slots__ = ("data",)

    @classmethod
    def from_allft(
        cls: Type[AllFTTypeVar], filename: Union[str, Path, BytesIO]
    ) -> AllFTTypeVar:
        allft = (
            pd.read_csv(
                filename,
                header=None,
                names=allft_fields,
                sep=";",
                dtype={
                    "aobt": str,
                    "iobt": str,
                    "cobt": str,
                    "eobt": str,
                    "arcAddr": str,
                },
                skiprows=1,
            )
            .assign(
                icao24=lambda df: df.arcAddr.str.lower(),
                aobt=lambda df: parse_date(df.aobt),
                iobt=lambda df: parse_date(df.iobt),
                cobt=lambda df: parse_date(df.cobt),
                eobt=lambda df: parse_date(df.eobt),
            )
            .rename(
                columns={
                    "departureAerodromeIcaoId": "origin",
                    "arrivalAerodromeIcaoId": "destination",
                    "aircraftId": "callsign",
                    "aircraftTypeIcaoId": "typecode",
                    "aobt": "AOBT",
                    "iobt": "IOBT",
                    "cobt": "COBT",
                    "eobt": "EOBT",
                }
            )
        )

        return cls(allft.sort_values("EOBT"))

    @classmethod
    def from_allft_7z(
        cls: Type[AllFTTypeVar], filename: Union[str, Path]
    ) -> AllFTTypeVar:
        from libarchive.public import memory_reader

        _prepare_libarchive()

        with open(filename, "rb") as fh:
            with memory_reader(fh.read()) as entries:
                b = BytesIO()
                for file in entries:
                    for block in file.get_blocks():
                        b.write(block)

        cumul = list()
        max_, current, previous = b.tell(), 0, 0
        b.seek(0)

        iterator = pd.read_csv(
            b,
            header=None,
            names=allft_fields,
            sep=";",
            dtype={
                "aobt": str,
                "iobt": str,
                "cobt": str,
                "eobt": str,
                "arcAddr": str,
            },
            skiprows=1,
            chunksize=2000,
        )

        with tqdm(
            total=max_, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in iterator:
                cumul.append(
                    chunk.assign(
                        icao24=lambda df: df.arcAddr.str.lower(),
                        aobt=lambda df: parse_date(df.aobt),
                        iobt=lambda df: parse_date(df.iobt),
                        cobt=lambda df: parse_date(df.cobt),
                        eobt=lambda df: parse_date(df.eobt),
                    ).rename(
                        columns={
                            "departureAerodromeIcaoId": "origin",
                            "arrivalAerodromeIcaoId": "destination",
                            "aircraftId": "callsign",
                            "aobt": "AOBT",
                            "iobt": "IOBT",
                            "cobt": "COBT",
                            "eobt": "EOBT",
                        }
                    )
                )

                current = b.tell()
                if current != previous:
                    pbar.update(current - previous)
                previous = current

        return cls(pd.concat(cumul, sort=False).sort_values("EOBT"))

    @classmethod
    def from_file(
        cls: Type[AllFTTypeVar], filename: Union[Path, str], **kwargs
    ) -> Optional[AllFTTypeVar]:  # coverage: ignore
        """
        In addition to `usual formats
        <export.html#traffic.core.mixins.DataFrameMixin>`_, you can parse so6
        files as text files (.ALL_FT+ extension) or as 7-zipped text files
        (.ALL_FT+.7z extension).

        .. warning::

            You will need the `libarchive
            <https://github.com/dsoprea/PyEasyArchive>`_ library to be able
            to parse .ALL_FT+.7z files on the fly.

        """
        path = Path(filename)
        if path.suffixes == [".ALL_FT+", ".7z"]:
            return cls.from_allft_7z(filename)
        if path.suffixes == [".ALL_FT+"]:
            return cls.from_allft(filename)
        return super().from_file(filename)

    def _repr_html_(self):
        return (
            self.data[
                [
                    "origin",
                    "destination",
                    "callsign",
                    "icao24",
                    "AOBT",
                    "ifpsId",
                    "IOBT",
                    "COBT",
                    "EOBT",
                    "flightState",
                    "mostPenalizingRegulationId",
                ]
            ]
            .set_index("ifpsId")
            ._repr_html_()
        )

    def _ipython_key_completions_(self):
        return {
            *self.data.ifpsId.values,
            *self.data.callsign.values,
            *self.data.icao24.values,
        }

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(
        self, item: Union[str, Iterable[str]]
    ) -> Union[None, "AllFT", FlightInfo]:
        res: Optional["AllFT"] = self
        if isinstance(item, str):
            if item in self.data.ifpsId.values:
                res = self.query(f'ifpsId == "{item}"')
                return FlightInfo(res.data.iloc[0]) if res is not None else None
            if item in self.data.icao24.values:
                return self.query(f'icao24 == "{item}"')
            if item in self.data.callsign.values:
                return self.query(f'callsign == "{item}"')
            return None
        if isinstance(item, Iterable):
            origin, destination = item
            if res is not None and origin is not None:
                res = res.query(f'origin == "{origin}"')
            if res is not None and destination is not None:
                res = res.query(f'destination == "{destination}"')
            return res
        return None

    def __iter__(self) -> Iterator[FlightInfo]:
        for _, elt in self.data.iterrows():
            yield FlightInfo(elt)
