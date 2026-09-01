# ruff: noqa: E501

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import httpx

import pandas as pd

from .... import cache_expiration
from ... import client
from ..models import EntityKind, ResolutionCandidate

_NASR_ANCHOR = date(2026, 2, 19)
_NASR_BASE_URL = "https://nfdc.faa.gov/webContent/28DaySub"
NasrPolicy = Literal[
    "auto_current",
    "latest_cached_only",
    "effective_date",
]


def _current_airac_effective_date(today: None | date = None) -> date:
    if today is None:
        today = datetime.now(timezone.utc).date()
    delta_days = (today - _NASR_ANCHOR).days
    cycles = delta_days // 28
    return _NASR_ANCHOR + timedelta(days=28 * cycles)


def _nasr_url_for_effective_date(effective: date) -> str:
    return (
        f"{_NASR_BASE_URL}/"
        f"28DaySubscription_Effective_{effective.strftime('%Y-%m-%d')}.zip"
    )


def _parse_effective_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _pick_existing_nasr_zip(folder: Path) -> None | Path:
    candidates = list(folder.glob("28DaySubscription_Effective_*.zip"))
    if len(candidates) == 0:
        candidates = list(folder.glob("NASR_*.zip"))
    if len(candidates) == 0:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _download_nasr_zip(folder: Path, target: Path) -> Path:
    effective = _current_airac_effective_date()
    dates = [
        effective,
        effective + timedelta(days=28),
        effective - timedelta(days=28),
    ]
    last_error: None | Exception = None

    for day in dates:
        url = _nasr_url_for_effective_date(day)
        try:
            response = client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)
            return target
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            "Unable to download current NASR cycle"
        ) from last_error
    raise RuntimeError("Unable to download current NASR cycle")


def _download_exact_nasr_zip(target: Path, effective: date) -> Path:
    response = client.get(_nasr_url_for_effective_date(effective))
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def _resolve_nasr_zip_path(
    path: str | Path,
    *,
    policy: NasrPolicy,
    effective_date: None | date,
) -> Path:
    requested = Path(path).expanduser()

    if requested.exists() and requested.is_file():
        return requested

    if requested.suffix.lower() == ".zip":
        requested.parent.mkdir(parents=True, exist_ok=True)
        if requested.exists():
            return requested

        if policy == "latest_cached_only":
            raise FileNotFoundError(
                f"NASR zip does not exist: {requested} (policy=latest_cached_only)"
            )

        if policy == "effective_date":
            if effective_date is None:
                raise ValueError(
                    "effective_date policy requires effective_date=YYYY-MM-DD"
                )
            return _download_exact_nasr_zip(requested, effective_date)

        m = re.search(r"Effective_(\d{4}-\d{2}-\d{2})", requested.name)
        if m is not None:
            return _download_exact_nasr_zip(
                requested,
                date.fromisoformat(m.group(1)),
            )

        return _download_nasr_zip(requested.parent, requested)

    requested.mkdir(parents=True, exist_ok=True)
    existing = _pick_existing_nasr_zip(requested)
    if existing is not None:
        return existing

    if policy == "latest_cached_only":
        raise FileNotFoundError(
            f"No NASR zip found in {requested} (policy=latest_cached_only)"
        )

    if policy == "effective_date":
        if effective_date is None:
            raise ValueError(
                "effective_date policy requires effective_date=YYYY-MM-DD"
            )
        target = requested / (
            f"28DaySubscription_Effective_{effective_date:%Y-%m-%d}.zip"
        )
        return _download_exact_nasr_zip(target, effective_date)

    effective = _current_airac_effective_date()
    target = requested / f"28DaySubscription_Effective_{effective:%Y-%m-%d}.zip"
    return _download_nasr_zip(requested, target)


class NasrAirportsProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        policy: NasrPolicy = "auto_current",
        effective_date: None | str | date = None,
    ) -> None:
        self.requested_path = Path(path)
        self.policy: NasrPolicy = policy
        self.effective_date = (
            None
            if effective_date is None
            else _parse_effective_date(effective_date)
        )
        self.source = "faa_nasr"
        self.priority = 100
        self.kinds = ("airport",)
        self.name = f"nasr_airports:{self.requested_path.as_posix()}"
        try:
            from thrust.airports import NasrAirportsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for NASR resolver source"
            ) from exc

        self._source_cls = NasrAirportsSource
        self._source: None | Any = None

    def _load_source(self) -> Any:
        if self._source is None:
            path = _resolve_nasr_zip_path(
                self.requested_path,
                policy=self.policy,
                effective_date=self.effective_date,
            )
            self._source = self._source_cls(path.as_posix())
        return self._source

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

        source = self._load_source()
        matches = source.resolve_airport(code.upper())
        return [
            ResolutionCandidate(
                code=(record.icao or record.iata or record.code),
                kind="airport",
                source=self.source,
                confidence=0.85,
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
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_airports()]
        return pd.DataFrame.from_records(rows)


class NasrNavpointsProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        policy: NasrPolicy = "auto_current",
        effective_date: None | str | date = None,
    ) -> None:
        self.requested_path = Path(path)
        self.policy: NasrPolicy = policy
        self.effective_date = (
            None
            if effective_date is None
            else _parse_effective_date(effective_date)
        )
        self.source = "faa_nasr"
        self.priority = 100
        self.kinds: tuple[str, ...] = ("fix", "navaid")
        self.name = f"nasr_navpoints:{self.requested_path.as_posix()}"
        try:
            from thrust.navpoints import NasrNavpointsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for NASR resolver source"
            ) from exc

        self._source_cls = NasrNavpointsSource
        self._source: None | Any = None

    def _load_source(self) -> Any:
        if self._source is None:
            path = _resolve_nasr_zip_path(
                self.requested_path,
                policy=self.policy,
                effective_date=self.effective_date,
            )
            self._source = self._source_cls(path.as_posix())
        return self._source

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

        source = self._load_source()
        matches = source.resolve_point(code.upper(), kind)
        return [
            ResolutionCandidate(
                code=record.code,
                kind=record.kind,
                source=self.source,
                confidence=0.85,
                payload={
                    "name": record.name,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "identifier": record.identifier,
                    "point_type": record.point_type,
                    "description": record.description,
                    "frequency": record.frequency,
                    "region": record.region,
                },
            )
            for record in matches
        ]

    def to_dataframe(self, kind: None | str = None) -> pd.DataFrame:
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_points(kind)]
        return pd.DataFrame.from_records(rows)


class NasrAirwaysProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        policy: NasrPolicy = "auto_current",
        effective_date: None | str | date = None,
    ) -> None:
        self.requested_path = Path(path)
        self.policy: NasrPolicy = policy
        self.effective_date = (
            None
            if effective_date is None
            else _parse_effective_date(effective_date)
        )
        self.source = "faa_nasr"
        self.priority = 100
        self.kinds = ("airway",)
        self.name = f"nasr_airways:{self.requested_path.as_posix()}"
        try:
            from thrust.airways import NasrAirwaysSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for NASR resolver source"
            ) from exc

        self._source_cls = NasrAirwaysSource
        self._source: None | Any = None

    def _load_source(self) -> Any:
        if self._source is None:
            path = _resolve_nasr_zip_path(
                self.requested_path,
                policy=self.policy,
                effective_date=self.effective_date,
            )
            self._source = self._source_cls(path.as_posix())
        return self._source

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

        source = self._load_source()
        matches = source.resolve_airway(code.upper())
        return [
            ResolutionCandidate(
                code=record.name,
                kind="airway",
                source=self.source,
                confidence=0.85,
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
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_airways()]
        return pd.DataFrame.from_records(rows)


class NasrAirspacesProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        policy: NasrPolicy = "auto_current",
        effective_date: None | str | date = None,
    ) -> None:
        self.requested_path = Path(path)
        self.policy: NasrPolicy = policy
        self.effective_date = (
            None
            if effective_date is None
            else _parse_effective_date(effective_date)
        )
        self.source = "faa_nasr"
        self.priority = 100
        self.kinds = ("airspace",)
        self.name = f"nasr_airspaces:{self.requested_path.as_posix()}"
        try:
            from thrust.airspaces import NasrAirspacesSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for NASR airspace source"
            ) from exc

        self._source_cls = NasrAirspacesSource
        self._source: None | Any = None

    def _load_source(self) -> Any:
        if self._source is None:
            path = _resolve_nasr_zip_path(
                self.requested_path,
                policy=self.policy,
                effective_date=self.effective_date,
            )
            self._source = self._source_cls(path.as_posix())
        return self._source

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

        source = self._load_source()
        matches = source.resolve_airspace(code.upper())
        return [
            ResolutionCandidate(
                code=record.designator,
                kind="airspace",
                source=self.source,
                confidence=0.92,
                payload={
                    "designator": record.designator,
                    "name": record.name,
                    "type": record.type_,
                    "lower": record.lower,
                    "upper": record.upper,
                    "coordinates": record.coordinates,
                },
            )
            for record in matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_airspaces()]
        return pd.DataFrame.from_records(rows)


class FaaArcgisAirspacesProvider:
    DATASETS: ClassVar[dict[str, str]] = {
        "faa_airports.json": "e747ab91a11045e8b3f8a3efd093d3b5_0",
        "faa_ats_routes.json": "acf64966af5f48a1a40fdbcb31238ba7_0",
        "faa_designated_points.json": "861043a88ff4486c97c3789e7dcdccc6_0",
        "faa_navaid_components.json": "c9254c171b6741d3a5e494860761443a_0",
        "faa_airspace_boundary.json": "67885972e4e940b2aa6d74024901c561_0",
        "faa_class_airspace.json": "c6a62360338e408cb1512366ad61559e_0",
        "faa_special_use_airspace.json": "dd0d1b726e504137ab3c41b21835d05b_0",
        "faa_route_airspace.json": "8bf861bb9b414f4ea9f0ff2ca0f1a851_0",
        "faa_prohibited_airspace.json": "354ee0c77484461198ebf728a2fca50c_0",
    }

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.source = "faa_arcgis"
        self.priority = 95
        self.kinds: tuple[str, ...] = ("airspace",)
        self.name = f"faa_arcgis_airspaces:{self.path.as_posix()}"
        try:
            from thrust.airspaces import FaaAirspacesSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for FAA ArcGIS airspace source"
            ) from exc

        self._source_cls = FaaAirspacesSource
        self._source: None | Any = None

    def _dataset_url(self, dataset_id: str) -> str:
        return f"https://opendata.arcgis.com/datasets/{dataset_id}.geojson"

    def _needs_refresh(self, target: Path) -> bool:
        if not target.exists():
            return True
        if cache_expiration is None:
            return False
        last_modification = target.lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        return bool(delta > cache_expiration)

    def _download_dataset(self, dataset_id: str, target: Path) -> None:
        response = client.get(self._dataset_url(dataset_id))
        response.raise_for_status()
        with target.open("w") as fh:
            json.dump(response.json(), fh)

    def _ensure_cache(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        for filename, dataset_id in self.DATASETS.items():
            target = self.path / filename
            if self._needs_refresh(target):
                try:
                    self._download_dataset(dataset_id, target)
                except httpx.TransportError:
                    if not target.exists():
                        raise

    def _load_source(self) -> Any:
        if self._source is None:
            self._ensure_cache()
            self._source = self._source_cls(self.path.as_posix())
        return self._source

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

        source = self._load_source()
        matches = source.resolve_airspace(code.upper())
        return [
            ResolutionCandidate(
                code=record.designator,
                kind="airspace",
                source=self.source,
                confidence=0.9,
                payload={
                    "designator": record.designator,
                    "name": record.name,
                    "type": record.type_,
                    "lower": record.lower,
                    "upper": record.upper,
                    "coordinates": record.coordinates,
                },
            )
            for record in matches
        ]

    def to_dataframe(self) -> pd.DataFrame:
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_airspaces()]
        return pd.DataFrame.from_records(rows)


class FaaArcgisNavpointsProvider(FaaArcgisAirspacesProvider):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.source = "faa_arcgis"
        self.priority = 95
        self.kinds: tuple[str, ...] = ("fix", "navaid")
        self.name = f"faa_arcgis_navpoints:{self.path.as_posix()}"
        try:
            from thrust.navpoints import FaaArcgisNavpointsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for FAA ArcGIS navpoint source"
            ) from exc

        self._source_cls = FaaArcgisNavpointsSource
        self._source: None | Any = None

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

        source = self._load_source()
        matches = source.resolve_point(code.upper(), kind)
        return [
            ResolutionCandidate(
                code=record.code,
                kind=record.kind,
                source=self.source,
                confidence=0.88,
                payload={
                    "name": record.name,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "identifier": record.identifier,
                    "point_type": record.point_type,
                    "description": record.description,
                    "frequency": record.frequency,
                    "region": record.region,
                },
            )
            for record in matches
        ]

    def to_dataframe(self, kind: None | str = None) -> pd.DataFrame:
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_points(kind)]
        return pd.DataFrame.from_records(rows)


class FaaArcgisAirwaysProvider(FaaArcgisAirspacesProvider):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.source = "faa_arcgis"
        self.priority = 95
        self.kinds = ("airway",)
        self.name = f"faa_arcgis_airways:{self.path.as_posix()}"
        try:
            from thrust.airways import FaaArcgisAirwaysSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for FAA ArcGIS airway source"
            ) from exc

        self._source_cls = FaaArcgisAirwaysSource
        self._source: None | Any = None

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

        source = self._load_source()
        matches = source.resolve_airway(code.upper())
        return [
            ResolutionCandidate(
                code=record.name,
                kind="airway",
                source=self.source,
                confidence=0.88,
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
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_airways()]
        return pd.DataFrame.from_records(rows)


class FaaArcgisAirportsProvider(FaaArcgisAirspacesProvider):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.source = "faa_arcgis"
        self.priority = 95
        self.kinds = ("airport",)
        self.name = f"faa_arcgis_airports:{self.path.as_posix()}"
        try:
            from thrust.airports import FaaArcgisAirportsSource
        except ImportError as exc:  # coverage: ignore
            raise RuntimeError(
                "thrust Python bindings are required for FAA ArcGIS airport source"
            ) from exc

        self._source_cls = FaaArcgisAirportsSource
        self._source: None | Any = None

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

        source = self._load_source()
        matches = source.resolve_airport(code.upper())
        return [
            ResolutionCandidate(
                code=(record.icao or record.iata or record.code),
                kind="airport",
                source=self.source,
                confidence=0.88,
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
        source = self._load_source()
        rows = [record.to_dict() for record in source.list_airports()]
        return pd.DataFrame.from_records(rows)
