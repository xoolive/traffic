from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

EntityKind = Literal[
    "airport",
    "navaid",
    "fix",
    "airway",
    "sid",
    "star",
    "airspace",
]


@dataclass(frozen=True)
class ResolverSource:
    family: Literal["eurocontrol", "faa", "other"]
    label: str
    path: None | str = None
    airac: None | str = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolutionCandidate:
    code: str
    kind: EntityKind
    source: str
    confidence: float
    valid_from: None | pd.Timestamp = None
    valid_to: None | pd.Timestamp = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolutionResult:
    query: str
    kind: None | EntityKind
    selected: None | ResolutionCandidate
    alternatives: tuple[ResolutionCandidate, ...] = tuple()
    reason: None | str = None


class ResolverProvider(Protocol):
    source: str
    name: str
    priority: int

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]: ...


def row_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            payload[key] = None
        else:
            payload[key] = value
    return payload
