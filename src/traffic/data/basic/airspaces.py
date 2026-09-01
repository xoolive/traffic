from __future__ import annotations

from typing import Any

import pandas as pd
from shapely.geometry import Polygon

from ...core.airspace import Airspace, ExtrudedPolygon, unary_union_with_alt


class Airspaces:
    def __init__(
        self,
        source: None | str = None,
        airspace_type: None | str = None,
        search_text: None | str = None,
    ) -> None:
        self._resolver_source = source
        self._airspace_type = airspace_type
        self._search_text = search_text

    def _clone(
        self,
        *,
        source: None | str = None,
        airspace_type: None | str = None,
        search_text: None | str = None,
    ) -> "Airspaces":
        return Airspaces(
            source=self._resolver_source if source is None else source,
            airspace_type=(
                self._airspace_type if airspace_type is None else airspace_type
            ),
            search_text=(
                self._search_text if search_text is None else search_text
            ),
        )

    def source(self, source: str) -> "Airspaces":
        return self._clone(source=source)

    def type(self, name: str) -> "Airspaces":
        return self._clone(airspace_type=name)

    @property
    def fra(self) -> "Airspaces":
        return self._clone(airspace_type="fra")

    def search(self, name: str) -> "Airspaces":
        return self._clone(search_text=name)

    @staticmethod
    def _type_mask(frame: pd.DataFrame, name: str) -> pd.Series:
        query = name.strip().upper()
        if "type" in frame.columns:
            return (
                frame["type"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(query, regex=False)
            )
        if "name" in frame.columns:
            return (
                frame["name"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(query, regex=False)
            )
        return pd.Series(False, index=frame.index)

    @staticmethod
    def _search_mask(frame: pd.DataFrame, name: str) -> pd.Series:
        query = name.strip().upper()
        designator_mask = pd.Series(False, index=frame.index)
        name_mask = pd.Series(False, index=frame.index)

        if "designator" in frame.columns:
            designator_mask = (
                frame["designator"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(query, regex=False)
            )

        if "name" in frame.columns:
            name_mask = (
                frame["name"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.contains(query, regex=False)
            )

        return designator_mask | name_mask

    def _payload_matches_filters(self, payload: dict[str, Any]) -> bool:
        if self._airspace_type is not None:
            candidate_type = str(payload.get("type") or "").upper()
            expected = self._airspace_type.strip().upper()
            candidate_name = str(payload.get("name") or "").upper()
            if (
                expected not in candidate_type
                and expected not in candidate_name
            ):
                return False

        if self._search_text is not None:
            query = self._search_text.strip().upper()
            designator = str(payload.get("designator") or "").upper()
            name = str(payload.get("name") or "").upper()
            if query not in designator and query not in name:
                return False

        return True

    @property
    def data(self) -> pd.DataFrame:
        from .. import resolver

        selected = self._resolver_source or "eurocontrol_aixm"
        frame = resolver.data(source=selected, kind="airspace")
        if frame.empty:
            return frame
        if self._airspace_type is not None:
            frame = frame.loc[self._type_mask(frame, self._airspace_type)]
        if self._search_text is not None:
            frame = frame.loc[self._search_mask(frame, self._search_text)]
        return frame

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.get(key)
        return self.data.__getitem__(key)

    def get(
        self,
        name: str,
        source: None | str = None,
        **kwargs: Any,
    ) -> Airspace:
        from .. import resolver

        selected_source = (
            source if source is not None else self._resolver_source
        )
        result = resolver.resolve(
            airspace=name,
            source=selected_source,
            **kwargs,
        )
        if result.selected is None:
            raise AttributeError(f"Airspace {name} not found")

        if not self._payload_matches_filters(result.selected.payload):
            raise AttributeError(f"Airspace {name} not found")

        candidate = result.selected
        candidates = [candidate, *list(result.alternatives)]

        elements = []
        for item in candidates:
            payload = item.payload
            layers = payload.get("layers") or []

            if isinstance(layers, list) and len(layers) > 0:
                for layer in layers:
                    if not isinstance(layer, dict):
                        continue
                    coords = layer.get("coordinates") or []
                    if len(coords) < 3:
                        continue
                    polygon = Polygon(coords)
                    if not polygon.is_valid:
                        continue
                    elements.append(
                        ExtrudedPolygon(
                            polygon,
                            layer.get("lower"),
                            layer.get("upper"),
                        )
                    )
                continue

            coords = payload.get("coordinates") or []
            if len(coords) < 3:
                continue
            polygon = Polygon(coords)
            if not polygon.is_valid:
                continue
            elements.append(
                ExtrudedPolygon(
                    polygon,
                    payload.get("lower"),
                    payload.get("upper"),
                )
            )

        elements = unary_union_with_alt(elements)

        if len(elements) == 0:
            raise AttributeError(f"Airspace {name} has no polygon data")

        return Airspace(
            name=str(candidate.payload.get("name") or candidate.code),
            designator=str(
                candidate.payload.get("designator") or candidate.code
            ),
            type_=candidate.payload.get("type"),
            elements=elements,
        )

    def free_route_areas(
        self,
        source: None | str = None,
    ) -> pd.DataFrame:
        selected = self if source is None else self.source(source)
        return selected.fra.data
