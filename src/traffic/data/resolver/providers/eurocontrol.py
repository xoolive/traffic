# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..models import EntityKind, ResolutionCandidate


class AIXMAirportsProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_aixm"
        self.priority = 100
        self.kinds = ("airport",)
        self.name = f"aixm_airports:{self.path.as_posix()}"
        try:
            from thrust.airports import AixmAirportsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for AIXM resolver source"
            ) from exc

        self._source = AixmAirportsSource(self.path.expanduser().as_posix())

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind is not None and kind != "airport":
            return []

        matches = self._source.resolve_airport(code.upper())
        return [
            ResolutionCandidate(
                code=(record.icao or record.iata or record.code),
                kind="airport",
                source=self.source,
                confidence=0.95,
                payload={
                    "name": record.name,
                    "iata": record.iata,
                    "icao": record.icao,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "altitude": record.altitude,
                    "country": record.country,
                },
            )
            for record in matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_airports()]
        return pd.DataFrame.from_records(rows)


class AIXMNavpointsProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_aixm"
        self.priority = 100
        self.kinds = ("fix", "navaid")
        self.name = f"aixm_navpoints:{self.path.as_posix()}"
        try:
            from thrust.navpoints import AixmNavpointsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for AIXM resolver source"
            ) from exc

        self._source = AixmNavpointsSource(self.path.expanduser().as_posix())

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "fix", "navaid"):
            return []

        matches = self._source.resolve_point(code.upper(), kind)
        return [
            ResolutionCandidate(
                code=record.code,
                kind=record.kind,
                source=self.source,
                confidence=0.95,
                payload={
                    "name": record.name,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "identifier": record.identifier,
                    "point_type": record.point_type,
                    "description": record.description,
                    "region": record.region,
                },
            )
            for record in matches
        ]

    def to_dataframe(self, kind: None | str = None) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_points(kind)]
        return pd.DataFrame.from_records(rows)


class AIXMAirwaysProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_aixm"
        self.priority = 100
        self.kinds = ("airway",)
        self.name = f"aixm_airways:{self.path.as_posix()}"
        try:
            from thrust.airways import AixmAirwaysSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for AIXM resolver source"
            ) from exc

        self._source = AixmAirwaysSource(self.path.expanduser().as_posix())

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "airway"):
            return []

        matches = self._source.resolve_airway(code.upper())
        return [
            ResolutionCandidate(
                code=record.name,
                kind="airway",
                source=self.source,
                confidence=0.95,
                payload={
                    "name": record.name,
                    "points": [
                        {
                            "code": point.code,
                            "raw_code": point.raw_code,
                            "kind": point.kind,
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                        }
                        for point in record.points
                    ],
                },
            )
            for record in matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["name", "points", "source"])


class AIXMAirspacesProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_aixm"
        self.priority = 100
        self.kinds = ("airspace",)
        self.name = f"aixm_airspaces:{self.path.as_posix()}"
        try:
            from thrust.airspaces import (
                AixmAirspacesSource,
                AixmFraAirspacesSource,
            )
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for AIXM resolver source"
            ) from exc

        self._source = AixmAirspacesSource(self.path.expanduser().as_posix())
        self._fra_source = AixmFraAirspacesSource(
            self.path.expanduser().as_posix()
        )

    @staticmethod
    def _record_to_candidate(
        record: object,
        source: str,
    ) -> ResolutionCandidate:
        return ResolutionCandidate(
            code=getattr(record, "designator"),
            kind="airspace",
            source=source,
            confidence=0.95,
            payload={
                "designator": getattr(record, "designator"),
                "name": getattr(record, "name"),
                "type": getattr(record, "type_"),
                "lower": getattr(record, "lower"),
                "upper": getattr(record, "upper"),
                "coordinates": getattr(record, "coordinates"),
            },
        )

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "airspace"):
            return []

        base_matches = self._source.resolve_airspace(code.upper())
        fra_matches = self._fra_source.resolve_airspace(code.upper())
        all_matches = [
            *base_matches,
            *fra_matches,
        ]
        return [
            self._record_to_candidate(record, self.source)
            for record in all_matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_airspaces()]
        fra_rows = [
            record.to_dict() for record in self._fra_source.list_airspaces()
        ]
        frame = pd.DataFrame.from_records([*rows, *fra_rows])
        if frame.empty:
            return frame
        return frame.drop_duplicates(
            subset=["designator", "name", "type", "lower", "upper"]
        )


class DDRAirspacesProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_ddr"
        self.priority = 90
        self.kinds = ("airspace",)
        self.name = f"ddr_airspaces:{self.path.as_posix()}"
        try:
            from thrust.airspaces import (
                DdrAirspacesSource,
                DdrFraAirspacesSource,
            )
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for DDR resolver source"
            ) from exc

        self._source = DdrAirspacesSource(self.path.expanduser().as_posix())
        self._fra_source = DdrFraAirspacesSource(
            self.path.expanduser().as_posix()
        )

    @staticmethod
    def _record_to_candidate(
        record: object,
        source: str,
    ) -> ResolutionCandidate:
        return ResolutionCandidate(
            code=getattr(record, "designator"),
            kind="airspace",
            source=source,
            confidence=0.9,
            payload={
                "designator": getattr(record, "designator"),
                "name": getattr(record, "name"),
                "type": getattr(record, "type_"),
                "lower": getattr(record, "lower"),
                "upper": getattr(record, "upper"),
                "coordinates": getattr(record, "coordinates"),
            },
        )

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "airspace"):
            return []

        base_matches = self._source.resolve_airspace(code.upper())
        fra_matches = self._fra_source.resolve_airspace(code.upper())
        all_matches = [
            *base_matches,
            *fra_matches,
        ]
        return [
            self._record_to_candidate(record, self.source)
            for record in all_matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_airspaces()]
        fra_rows = [
            record.to_dict() for record in self._fra_source.list_airspaces()
        ]
        frame = pd.DataFrame.from_records([*rows, *fra_rows])
        if frame.empty:
            return frame
        return frame.drop_duplicates(
            subset=["designator", "name", "type", "lower", "upper"]
        )


class DDRNavpointsProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_ddr"
        self.priority = 90
        self.kinds = ("fix", "navaid")
        self.name = f"ddr_navpoints:{self.path.as_posix()}"
        try:
            from thrust.navpoints import DdrNavpointsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for DDR resolver source"
            ) from exc

        self._source = DdrNavpointsSource(self.path.expanduser().as_posix())

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "fix", "navaid"):
            return []

        matches = self._source.resolve_point(code.upper(), kind)
        return [
            ResolutionCandidate(
                code=record.code,
                kind=record.kind,
                source=self.source,
                confidence=0.9,
                payload={
                    "name": record.name,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "identifier": record.identifier,
                    "point_type": record.point_type,
                    "description": record.description,
                    "region": record.region,
                },
            )
            for record in matches
        ]

    def to_dataframe(self, kind: None | str = None) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_points(kind)]
        return pd.DataFrame.from_records(rows)


class DDRAirwaysProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_ddr"
        self.priority = 90
        self.kinds = ("airway",)
        self.name = f"ddr_airways:{self.path.as_posix()}"
        try:
            from thrust.airways import DdrAirwaysSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for DDR resolver source"
            ) from exc

        self._source = DdrAirwaysSource(self.path.expanduser().as_posix())

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "airway"):
            return []

        matches = self._source.resolve_airway(code.upper())
        return [
            ResolutionCandidate(
                code=record.name,
                kind="airway",
                source=self.source,
                confidence=0.9,
                payload={
                    "name": record.name,
                    "points": [
                        {
                            "code": point.code,
                            "raw_code": point.raw_code,
                            "kind": point.kind,
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                        }
                        for point in record.points
                    ],
                },
            )
            for record in matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_airways()]
        return pd.DataFrame.from_records(rows)


class AIXMField15Provider:
    """Route enrichment provider backed by the AIXM AirwayDatabase (thrust Rust backend).

    Unlike the point-resolution providers, this provider takes a complete ICAO field 15
    route string and returns a list of resolved geographic segments.  It does not implement
    the ``ResolverProvider`` protocol (it has no ``resolve(code)`` method) and is used
    exclusively via ``Resolver.parse_route()``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_aixm"
        self.name = f"aixm_field15:{self.path.as_posix()}"
        self.priority = 100
        try:
            from thrust.core.field15 import AiracDatabase
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for AIXM field15 route enrichment"
            ) from exc

        self._db = AiracDatabase(self.path.expanduser().as_posix())

    def parse_route(self, route: str) -> list[dict[str, Any]]:
        """Parse and resolve a raw ICAO field 15 route string.

        Returns a list of segment dicts, each with keys:
        - ``start``: ``{latitude, longitude, name?}``
        - ``end``:   ``{latitude, longitude, name?}``
        - ``name``:  airway or connector name, or ``None`` for direct routings
        """
        segments = self._db.enrich_route(route)
        return [seg.to_dict() for seg in segments]


class DDRAirportsProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.source = "eurocontrol_ddr"
        self.priority = 90
        self.kinds = ("airport",)
        self.name = f"ddr_airports:{self.path.as_posix()}"
        try:
            from thrust.airports import DdrAirportsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for DDR airport source"
            ) from exc

        self._source = DdrAirportsSource(self.path.expanduser().as_posix())

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del reference
        del when

        if kind not in (None, "airport"):
            return []

        matches = self._source.resolve_airport(code.upper())
        return [
            ResolutionCandidate(
                code=(record.icao or record.iata or record.code),
                kind="airport",
                source=self.source,
                confidence=0.9,
                payload={
                    "name": record.name,
                    "iata": record.iata,
                    "icao": record.icao,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "altitude": record.altitude,
                    "country": record.country,
                },
            )
            for record in matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [record.to_dict() for record in self._source.list_airports()]
        return pd.DataFrame.from_records(rows)
