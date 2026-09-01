# ruff: noqa: E501

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .models import (
    EntityKind,
    ResolutionResult,
    ResolverProvider,
    ResolverSource,
)
from .providers import (
    AIXMAirportsProvider,
    AIXMAirspacesProvider,
    AIXMAirwaysProvider,
    AIXMField15Provider,
    AIXMNavpointsProvider,
    BasicAirwaysProvider,
    BasicNavaidsProvider,
    DDRAirportsProvider,
    DDRAirspacesProvider,
    DDRAirwaysProvider,
    DDRNavpointsProvider,
    FaaArcgisAirportsProvider,
    FaaArcgisAirspacesProvider,
    FaaArcgisAirwaysProvider,
    FaaArcgisNavpointsProvider,
    Fr24AirportsProvider,
    NasrAirportsProvider,
    NasrAirspacesProvider,
    NasrAirwaysProvider,
    NasrNavpointsProvider,
    NasrPolicy,
    OSMBeaconsProvider,
    OurAirportsProvider,
)

logger = logging.getLogger(__name__)


def _normalize_path(path: None | str | Path) -> None | str:
    if path is None:
        return None
    return Path(path).expanduser().as_posix()


class Resolver:
    def __init__(self, providers: None | list[ResolverProvider] = None) -> None:
        self._providers = providers or [
            Fr24AirportsProvider(),
            OurAirportsProvider(),
            BasicNavaidsProvider(),
            BasicAirwaysProvider(),
            OSMBeaconsProvider(),
        ]
        self._sources: list[ResolverSource] = [
            ResolverSource(family="other", label="flightradar24"),
            ResolverSource(family="other", label="ourairports"),
            ResolverSource(family="other", label="xplane"),
            ResolverSource(family="other", label="openstreetmap"),
        ]
        self._field15_provider: None | AIXMField15Provider = None

    def register(self, provider: ResolverProvider) -> "Resolver":
        self._providers = [
            p for p in self._providers if p.name != provider.name
        ]
        self._providers.append(provider)
        return self

    def eurocontrol(
        self,
        *,
        aixm: None | str | Path = None,
        ddr: None | str | Path = None,
        airac: None | str = None,
        **metadata: Any,
    ) -> "Resolver":
        if aixm is not None:
            normalized = _normalize_path(aixm)
            self._sources.append(
                ResolverSource(
                    family="eurocontrol",
                    label="eurocontrol_aixm",
                    path=normalized,
                    airac=airac,
                    metadata=metadata,
                )
            )
            if normalized is not None:
                self.register(AIXMAirportsProvider(normalized))
                self.register(AIXMNavpointsProvider(normalized))
                self.register(AIXMAirwaysProvider(normalized))
                self._field15_provider = AIXMField15Provider(normalized)
        if ddr is not None:
            normalized_ddr = _normalize_path(ddr)
            self._sources.append(
                ResolverSource(
                    family="eurocontrol",
                    label="eurocontrol_ddr",
                    path=normalized_ddr,
                    airac=airac,
                    metadata=metadata,
                )
            )
            if normalized_ddr is not None:
                self.register(DDRAirportsProvider(normalized_ddr))
                self.register(DDRAirspacesProvider(normalized_ddr))
                self.register(DDRNavpointsProvider(normalized_ddr))
                self.register(DDRAirwaysProvider(normalized_ddr))

        if aixm is not None:
            normalized = _normalize_path(aixm)
            if normalized is not None:
                self.register(AIXMAirspacesProvider(normalized))
        return self

    def faa(
        self,
        *,
        nasr: None | str | Path = None,
        arcgis: None | str | Path = None,
        airac: None | str = None,
        nasr_policy: NasrPolicy = "auto_current",
        effective_date: None | str = None,
        **metadata: Any,
    ) -> "Resolver":
        if nasr is not None:
            normalized = _normalize_path(nasr)
            self._sources.append(
                ResolverSource(
                    family="faa",
                    label="faa_nasr",
                    path=normalized,
                    airac=airac,
                    metadata=metadata,
                )
            )
            if normalized is not None:
                self.register(
                    NasrAirportsProvider(
                        normalized,
                        policy=nasr_policy,
                        effective_date=effective_date,
                    )
                )
                self.register(
                    NasrNavpointsProvider(
                        normalized,
                        policy=nasr_policy,
                        effective_date=effective_date,
                    )
                )
                self.register(
                    NasrAirwaysProvider(
                        normalized,
                        policy=nasr_policy,
                        effective_date=effective_date,
                    )
                )
                self.register(
                    NasrAirspacesProvider(
                        normalized,
                        policy=nasr_policy,
                        effective_date=effective_date,
                    )
                )
        if arcgis is not None:
            normalized_arcgis = _normalize_path(arcgis)
            if normalized_arcgis is None:
                from ... import cache_path

                normalized_arcgis = (cache_path / "faa_arcgis").as_posix()
            self._sources.append(
                ResolverSource(
                    family="faa",
                    label="faa_arcgis",
                    path=normalized_arcgis,
                    airac=airac,
                    metadata=metadata,
                )
            )
            self.register(FaaArcgisAirspacesProvider(normalized_arcgis))
            self.register(FaaArcgisAirportsProvider(normalized_arcgis))
            self.register(FaaArcgisNavpointsProvider(normalized_arcgis))
            self.register(FaaArcgisAirwaysProvider(normalized_arcgis))
        return self

    @property
    def sources(self) -> tuple[ResolverSource, ...]:
        return tuple(self._sources)

    def data(
        self,
        *,
        source: str,
        kind: None | EntityKind = None,
    ) -> pd.DataFrame:
        providers = [p for p in self._providers if p.source == source]
        if kind is not None:
            providers = [
                p for p in providers if kind in getattr(p, "kinds", tuple())
            ]

        for provider in providers:
            to_dataframe = getattr(provider, "to_dataframe", None)
            if callable(to_dataframe):
                try:
                    frame = to_dataframe(kind=kind)
                except TypeError:
                    frame = to_dataframe()
                if isinstance(frame, pd.DataFrame):
                    return frame
            frame = getattr(provider, "data", None)
            if isinstance(frame, pd.DataFrame):
                return frame

        return pd.DataFrame()

    def sources_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(source) for source in self._sources)

    def resolve(
        self,
        code: None | str = None,
        *,
        airport: None | str = None,
        navaid: None | str = None,
        fix: None | str = None,
        airway: None | str = None,
        airspace: None | str = None,
        source: None | str = None,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
        **kwargs: object,
    ) -> ResolutionResult:
        del kwargs
        query: str | None = None
        if airport is not None:
            query = airport
            kind = "airport"
        elif navaid is not None:
            query = navaid
            kind = "navaid"
        elif fix is not None:
            query = fix
            kind = "fix"
        elif airway is not None:
            query = airway
            kind = "airway"
        elif airspace is not None:
            query = airspace
            kind = "airspace"
        else:
            query = code
        if query is None:
            raise ValueError(
                "Pass code=..., airport=..., navaid=..., fix=..., airway=... or airspace=..."
            )

        candidates = []
        provider_errors: list[str] = []
        for provider in sorted(
            self._providers, key=lambda p: p.priority, reverse=True
        ):
            if source is not None and provider.source != source:
                continue
            try:
                candidates.extend(
                    provider.resolve(
                        query,
                        kind=kind,
                        reference=reference,
                        when=when,
                    )
                )
            except Exception as exc:
                provider_errors.append(f"{provider.name}: {exc}")
                logger.warning(
                    "Resolver provider %s failed", provider.name, exc_info=True
                )

        candidates = sorted(
            candidates, key=lambda c: c.confidence, reverse=True
        )

        if len(candidates) == 0:
            return ResolutionResult(
                query=query,
                kind=kind,
                selected=None,
                alternatives=tuple(),
                reason=(
                    "; ".join(provider_errors)
                    if provider_errors
                    else (
                        f"No provider resolved the code in source {source}"
                        if source is not None
                        else "No provider resolved the code"
                    )
                ),
            )

        return ResolutionResult(
            query=query,
            kind=kind,
            selected=candidates[0],
            alternatives=tuple(candidates[1:]),
        )

    def parse_route(self, route: str) -> list[dict[str, Any]]:
        """Parse and resolve a raw ICAO field 15 route string into geographic segments.

        Returns a list of segment dicts.  Each segment has the form::

            {
                "start": {"latitude": float, "longitude": float, "name": str | None},
                "end":   {"latitude": float, "longitude": float, "name": str | None},
                "name":  str | None,  # airway name, or None for DCT legs
            }

        Requires an AIXM data source to be registered (via ``resolver.eurocontrol(aixm=...)``).
        Raises ``RuntimeError`` if no field15-capable provider has been configured.
        """
        if self._field15_provider is None:
            raise RuntimeError(
                "No field15 provider configured. "
                "Call resolver.eurocontrol(aixm=<path>) first."
            )
        return self._field15_provider.parse_route(route)
