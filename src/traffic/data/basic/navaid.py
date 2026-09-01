# ruff: noqa: E501

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Iterator

import pandas as pd

from ...core.mixins import GeoDBMixin
from ...core.structure import Navaid

_log = logging.getLogger(__name__)


class Navaids(GeoDBMixin):
    """
    `VOR <https://en.wikipedia.org/wiki/VHF_omnidirectional_range>`_, `DME
    <https://en.wikipedia.org/wiki/Distance_measuring_equipment>`_ and `NDB
    <https://en.wikipedia.org/wiki/Non-directional_beacon>`_ are short-range
    radio navigation systems for aircraft.  They help aircraft with a receiving
    unit to determine their position and stay on course.

    The first airways were designed after the locations of these beacons. Recent
    progress in GNSS systems helped define more positions by their latitudes and
    longitudes, referred to as FIX.

    - Read more about `navigational beacons
      <https://aerosavvy.com/navigation-name-nonsense/>`_ and how FIX names are
      decided.

    - Read more about `calibration <scenarios/calibration.html>`_ of such
      equipment.

    A (deprecated) database of world navigational beacons is available as:

    ```pycon
    >>> from traffic.data import navaids

    ```

    Any navigational beacon can be accessed by the bracket notation:

    ```pycon
    >>> navaids['NARAK']
    Navaid('NARAK', type='FIX', latitude=44.29527778, longitude=1.74888889)

    ```

    """

    cache_path: Path
    alternatives: ClassVar[dict[str, "Navaids"]] = dict()
    name: str = "default"
    priority: int = 0

    def __init__(self, data: None | pd.DataFrame = None) -> None:
        self._data: None | pd.DataFrame = data
        self._resolver_source: None | str = None
        if self.name not in Navaids.alternatives:
            Navaids.alternatives[self.name] = self

    def source(self, source: str) -> "Navaids":
        clone = self.__class__(self._data)
        clone._extent = self._extent
        clone._resolver_source = source
        return clone

    @property
    def available(self) -> bool:
        return True

    def parse_data(self) -> None:  # coverage: ignore
        navaids = []

        cache_file = Path(__file__).parent.parent / "navdata" / "earth_fix.dat"
        assert cache_file.exists()
        iter_lines = cache_file.open("rb")

        for line_bytes in iter_lines:
            line = line_bytes.decode(encoding="ascii", errors="ignore").strip()

            # Skip empty lines or comments
            if len(line) < 3 or line[0] == "#":
                continue

            # Start with valid 2 digit latitude -45. or 52.
            if not ((line[0] == "-" and line[3] == ".") or line[2] == "."):
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example line:
            #  30.580372 -094.384169 FAREL
            fields = line.split()
            navaids.append(
                Navaid(
                    fields[2],
                    "FIX",
                    float(fields[0]),
                    float(fields[1]),
                    float("nan"),
                    None,
                    None,
                    None,
                )
            )

        cache_file = Path(__file__).parent.parent / "navdata" / "earth_nav.dat"
        assert cache_file.exists()
        iter_lines = cache_file.open("rb")

        for line_bytes in iter_lines:
            line = line_bytes.decode(encoding="ascii", errors="ignore").strip()

            # Skip empty lines or comments
            if len(line) == 0 or line[0] == "#":
                continue

            # Data line => Process fields of this record, separated by a comma
            # Example lines:
            # 2  58.61466599  125.42666626 451   522  30  0.0 A   Aldan NDB
            # 3  31.26894444 -085.72630556 334 11120  40 -3.0 OZR CAIRNS VOR-DME
            # type    lat       lon        elev freq  ?   var id   desc
            #   0      1         2           3    4   5    6   7    8

            fields = line.split()

            # Valid line starts with integers
            if not fields[0].isdigit():
                continue  # Next line

            # Get code for type of navaid
            itype = int(fields[0])

            # Type names
            wptypedict = {
                2: "NDB",
                3: "VOR",
                4: "ILS",
                5: "LOC",
                6: "GS",
                7: "OM",
                8: "MM",
                9: "IM",
                12: "DME",
                13: "TACAN",
            }

            # Type code never larger than 20
            if itype not in list(wptypedict.keys()):
                continue  # Next line

            wptype = wptypedict[itype]

            # Select types to read
            if wptype not in ["NDB", "VOR", "ILS", "GS", "DME", "TACAN"]:
                continue  # Next line

            # Find description
            try:
                idesc = line.index(fields[7]) + len(fields[7])
                description: None | str = line[idesc:].strip().upper()
            except Exception:
                description = None

            navaids.append(
                Navaid(
                    fields[7],
                    wptype,
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3][1:])
                    if fields[3].startswith("0-")
                    else float(fields[3]),
                    float(fields[4])
                    if wptype == "NDB"
                    else float(fields[4]) / 100,
                    float(fields[6])
                    if wptype in ["VOR", "NDB", "ILS", "GS"]
                    else None,
                    description,
                )
            )

        self._data = pd.DataFrame(
            [
                {
                    "name": navaid.name,
                    "type": navaid.type,
                    "latitude": navaid.latitude,
                    "longitude": navaid.longitude,
                    "altitude": navaid.altitude,
                    "frequency": navaid.frequency,
                    "magnetic_variation": navaid.magnetic_variation,
                    "description": navaid.description,
                }
                for navaid in navaids
            ]
        )

        self._data.to_parquet(self.cache_path / "traffic_navaid.parquet")

    @property
    def data(self) -> pd.DataFrame:
        if self._resolver_source is not None:
            from .. import resolver

            nav = resolver.data(source=self._resolver_source, kind="navaid")
            fix = resolver.data(source=self._resolver_source, kind="fix")
            frame = pd.concat(
                [x for x in [nav, fix] if not x.empty],
                ignore_index=True,
            )
            if frame.empty:
                raise RuntimeError(
                    f"No navaid/fix data found for source {self._resolver_source}"
                )

            frame = frame.rename(
                columns={
                    "code": "name",
                    "kind": "type",
                }
            )
            if "type" in frame.columns:
                frame["type"] = frame["type"].str.upper()
            if "altitude" not in frame.columns:
                frame["altitude"] = 0.0
            if "frequency" not in frame.columns:
                frame["frequency"] = None
            if "magnetic_variation" not in frame.columns:
                frame["magnetic_variation"] = None
            if "description" not in frame.columns:
                frame["description"] = None
            return frame

        if self._data is not None:
            return self._data

        if not (self.cache_path / "traffic_navaid.parquet").exists():
            self.parse_data()
        else:
            _log.info("Loading navaid database")
            self._data = pd.read_parquet(
                self.cache_path / "traffic_navaid.parquet"
            )

        if self._data is not None:
            self._data = self._data.rename(
                columns=dict(alt="altitude", lat="latitude", lon="longitude")
            )
        assert self._data is not None
        return self._data

    @lru_cache()
    def __getitem__(self, key: str | tuple[str, tuple[float, float]]) -> Navaid:
        if isinstance(key, tuple):
            name, reference = key
        else:
            name = key
            reference = None
        return self.get(
            name,
            reference=reference,
            source=self._resolver_source,
        )

    def global_get(self, name: str) -> Navaid:
        _log.warning("Use .get() function instead", DeprecationWarning)
        return self.get(name)

    def get(
        self,
        name: str,
        reference: None | tuple[float, float] = None,
        source: None | str = None,
        kind: None | str = None,
        **kwargs: Any,
    ) -> Navaid:
        """Search for a navaid/fix through the resolver.

        ```pycon
        >>> from traffic.data import navaids

        ```

        ```pycon
        >>> navaids.get("ZUE")  # doctest: +SKIP
        Navaid('ZUE', type='NDB', latitude=30.9, longitude=20.068, altitude=0.0, description='ZUEITINA NDB', frequency='369.0kHz')

        ```

        ```pycon
        >>> navaids.extent("Switzerland").get("ZUE")  # doctest: +SKIP
        Navaid('ZUE', type='VOR', latitude=47.592, longitude=8.817, altitude=1730.0, description='ZURICH EAST VOR-DME', frequency='110.05MHz')

        ```
        """
        from .. import resolver

        selected_source = (
            source if source is not None else self._resolver_source
        )

        if kind in ("fix", "navaid"):
            result = resolver.resolve(
                source=selected_source,
                kind=kind,  # type: ignore[arg-type]
                code=name,
                reference=reference,
                **kwargs,
            )
        else:
            nav = resolver.resolve(
                navaid=name,
                source=selected_source,
                reference=reference,
                **kwargs,
            )
            fix = resolver.resolve(
                fix=name,
                source=selected_source,
                reference=reference,
                **kwargs,
            )
            candidates = [
                c for c in [nav.selected, fix.selected] if c is not None
            ]
            if len(candidates) == 0:
                result = nav
            else:
                best = sorted(
                    candidates, key=lambda c: c.confidence, reverse=True
                )[0]
                result = resolver.resolve(
                    code=best.code,
                    source=best.source,
                    kind=best.kind,
                )

        candidate = result.selected
        if candidate is None:
            raise AttributeError(f"Point {name} not found")

        payload = candidate.payload
        navaid_name = str(candidate.code)
        navaid_type = str(payload.get("point_type") or candidate.kind).upper()
        description = payload.get("description")
        if description is None and payload.get("region") is not None:
            description = f"{navaid_name} {payload.get('region')}"

        frequency = payload.get("frequency")
        frequency_float = float(frequency) if frequency is not None else None

        return Navaid(
            navaid_name,
            "FIX" if candidate.kind == "fix" else navaid_type,
            float(payload.get("latitude") or 0.0),
            float(payload.get("longitude") or 0.0),
            float(payload.get("altitude") or 0.0),
            frequency_float,
            None,
            str(description) if description is not None else None,
        )

    def get_fix(
        self,
        name: str,
        source: None | str = None,
        **kwargs: Any,
    ) -> Navaid:
        return self.get(name, source=source, kind="fix", **kwargs)

    def get_navaid(
        self,
        name: str,
        source: None | str = None,
        **kwargs: Any,
    ) -> Navaid:
        return self.get(name, source=source, kind="navaid", **kwargs)

    def __iter__(self) -> Iterator[Navaid]:
        for _, x in self.data.iterrows():
            yield Navaid(**x)

    def search(self, name: str) -> "Navaids":
        """
        Selects the subset of airways matching name in the name or description
        field.

        !!! warning
            The same name may match several navigational beacons in the world.
            Use the extent() method to limit the search to an area of interest.

        ```pycon
        >>> navaids.search("ZUE")  # doctest: +SKIP
          name   type   latitude   longitude   altitude   frequency   description
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          ZUE    NDB    30.9       20.07       0          369         ZUEITINA NDB
          ZUE    VOR    47.59      8.818       1730       110         ZURICH EAST VOR-DME
          ZUE    DME    47.59      8.818       1730       110         ZURICH EAST VOR-DME
        ```

        ```pycon
        >>> navaids.extent("Switzerland").search("ZUE")  # doctest: +SKIP
          name   type   latitude   longitude   altitude   frequency   description
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          ZUE    VOR    47.59      8.818       1730       110         ZURICH EAST VOR-DME
          ZUE    DME    47.59      8.818       1730       110         ZURICH EAST VOR-DME
        ```
        """
        return self.__class__(
            self.data.query(
                "description == @name.upper() or name == @name.upper()"
            )
        )
