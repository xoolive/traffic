# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .core import Resolver
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

__all__ = [
    "AIXMAirportsProvider",
    "AIXMAirspacesProvider",
    "AIXMAirwaysProvider",
    "AIXMField15Provider",
    "AIXMNavpointsProvider",
    "BasicAirwaysProvider",
    "BasicNavaidsProvider",
    "DDRAirportsProvider",
    "DDRAirspacesProvider",
    "DDRAirwaysProvider",
    "DDRNavpointsProvider",
    "EntityKind",
    "FaaArcgisAirportsProvider",
    "FaaArcgisAirspacesProvider",
    "FaaArcgisAirwaysProvider",
    "FaaArcgisNavpointsProvider",
    "Fr24AirportsProvider",
    "NasrAirportsProvider",
    "NasrAirspacesProvider",
    "NasrAirwaysProvider",
    "NasrNavpointsProvider",
    "NasrPolicy",
    "OSMBeaconsProvider",
    "OurAirportsProvider",
    "ResolutionResult",
    "Resolver",
    "ResolverProvider",
    "ResolverSource",
    "data",
    "eurocontrol",
    "faa",
    "parse_route",
    "register",
    "resolve",
    "sources_frame",
]


_DEFAULT_RESOLVER = Resolver()


def register(provider: ResolverProvider) -> Resolver:
    global _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = _DEFAULT_RESOLVER.register(provider)
    return _DEFAULT_RESOLVER


def eurocontrol(
    *,
    aixm: None | str | Path = None,
    ddr: None | str | Path = None,
    airac: None | str = None,
    **metadata: Any,
) -> Resolver:
    global _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = _DEFAULT_RESOLVER.eurocontrol(
        aixm=aixm,
        ddr=ddr,
        airac=airac,
        **metadata,
    )
    return _DEFAULT_RESOLVER


def faa(
    *,
    nasr: None | str | Path = None,
    arcgis: None | str | Path = None,
    airac: None | str = None,
    nasr_policy: NasrPolicy = "auto_current",
    effective_date: None | str = None,
    **metadata: Any,
) -> Resolver:
    global _DEFAULT_RESOLVER
    _DEFAULT_RESOLVER = _DEFAULT_RESOLVER.faa(
        nasr=nasr,
        arcgis=arcgis,
        airac=airac,
        nasr_policy=nasr_policy,
        effective_date=effective_date,
        **metadata,
    )
    return _DEFAULT_RESOLVER


def sources_frame() -> pd.DataFrame:
    return _DEFAULT_RESOLVER.sources_frame()


def data(*, source: str, kind: None | EntityKind = None) -> pd.DataFrame:
    return _DEFAULT_RESOLVER.data(source=source, kind=kind)


def resolve(
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
    return _DEFAULT_RESOLVER.resolve(
        code=code,
        airport=airport,
        navaid=navaid,
        fix=fix,
        airway=airway,
        airspace=airspace,
        source=source,
        kind=kind,
        reference=reference,
        when=when,
        **kwargs,
    )


def parse_route(route: str) -> list[dict[str, Any]]:
    """Parse and resolve a raw ICAO field 15 route string into geographic segments.

    Requires the default resolver to have an AIXM data source configured.
    Call ``eurocontrol(aixm=<path>)`` first.

    Example::

        from traffic.data.resolver import eurocontrol, parse_route

        eurocontrol(aixm="~/data/airac/2501")
        segments = parse_route("LFPG DCT LACOU UM184 VEBIT DCT LFLL")
        for seg in segments:
            print(seg["start"]["name"], "->", seg["end"]["name"], "via", seg.get("name"))
    """
    return _DEFAULT_RESOLVER.parse_route(route)
