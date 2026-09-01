from __future__ import annotations

from typing import Any

from ...core.structure import Navaid


class Fixes:
    def __init__(self, source: None | str = None) -> None:
        self._resolver_source = source

    def source(self, source: str) -> "Fixes":
        return Fixes(source=source)

    def __getitem__(self, key: str) -> Navaid:
        return self.get(key)

    def get(
        self,
        name: str,
        source: None | str = None,
        **kwargs: Any,
    ) -> Navaid:
        from .. import resolver

        selected_source = (
            source if source is not None else self._resolver_source
        )
        result = resolver.resolve(fix=name, source=selected_source, **kwargs)
        if result.selected is None:
            raise AttributeError(f"Fix {name} not found")

        payload = result.selected.payload
        return Navaid(
            str(payload.get("name") or result.selected.code),
            "FIX",
            float(payload.get("latitude") or 0.0),
            float(payload.get("longitude") or 0.0),
            float(payload.get("altitude") or 0.0),
            None,
            None,
            str(payload.get("description"))
            if payload.get("description") is not None
            else None,
        )
