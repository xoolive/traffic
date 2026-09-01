from __future__ import annotations

import pandas as pd

from ..models import EntityKind, ResolutionCandidate, row_payload


class BasicNavaidsProvider:
    def __init__(self) -> None:
        self.source = "xplane"
        self.name = "xplane_navaids"
        self.priority = 40
        self.kinds = ("fix", "navaid")

    @property
    def data(self) -> pd.DataFrame:
        from ... import navaids

        return navaids.data

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

        upper = code.upper()
        matches = self.data.query("description == @upper or name == @upper")

        out: list[ResolutionCandidate] = []
        for _, row in matches.iterrows():
            rtype = str(row.get("type") or "").upper()
            point_kind: EntityKind = "fix" if rtype == "FIX" else "navaid"
            if kind is not None and point_kind != kind:
                continue

            out.append(
                ResolutionCandidate(
                    code=str(row.get("name") or upper),
                    kind=point_kind,
                    source=self.source,
                    confidence=0.65,
                    payload=row_payload(row),
                )
            )

        return out


class BasicAirwaysProvider:
    def __init__(self) -> None:
        self.source = "xplane"
        self.name = "xplane_airways"
        self.priority = 40
        self.kinds = ("airway",)

    @property
    def data(self) -> pd.DataFrame:
        from ... import airways

        return airways.data

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

        upper = code.upper()
        route = self.data.query("route == @upper").sort_values("id")
        if route.empty:
            return []

        points = [
            {
                "code": str(row.get("navaid")),
                "latitude": float(row.get("latitude")),
                "longitude": float(row.get("longitude")),
            }
            for _, row in route.iterrows()
        ]

        return [
            ResolutionCandidate(
                code=upper,
                kind="airway",
                source=self.source,
                confidence=0.6,
                payload={"name": upper, "points": points},
            )
        ]
