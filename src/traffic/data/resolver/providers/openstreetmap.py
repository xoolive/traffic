from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

import pandas as pd

from .... import cache_expiration, cache_path
from ..models import EntityKind, ResolutionCandidate


class OSMBeaconsProvider:
    """Resolver provider backed by OpenStreetMap ``airmark=beacon`` nodes.

    Fetching goes through :func:`cartes.osm.beacons`, the same Overpass layer
    `traffic` already uses for airport structures (and that powers
    :attr:`Airport.beacons`). Results are cached by cartes (file cache keyed by
    the query string; pass an AIRAC-day ``date`` to bust it). Taxonomy
    normalisation (``beacon:type`` + ``localizer``/``glideslope`` → the
    traffic navaid types) happens here in Python.

    Resolution strategy (area-scoped, lazy, registered by default):

    - ``resolve(reference=...)`` scopes the query to a bounding box around the
      reference point (~2°) — cheap, cached, and disambiguates duplicate
      beacon names by proximity.
    - ``resolve(reference=None)`` returns ``[]`` and triggers **no network**:
      OSM only contributes when a scope is present, so the default source
      never causes a global Overpass pull on a bare ``navaids.get(name)``.
    - ``data`` (i.e. ``resolver.data(source="osm")`` / ``navaids.search``) is
      the **explicit global** path: a one-shot fetch, normalised and cached to
      parquet with the usual cache expiration. This is the opt-in global pull.

    OSM beacons complement the other sources: coverage is patchy but global,
    AIRAC-independent, and includes ILS loc/gs + OM/MM/IM markers that the
    x-plane path drops.
    """

    # Half-extent (degrees) of the bbox used for reference-scoped resolution.
    # ~2° ≈ 220 km, matching the previous around radius in spirit.
    reference_half_extent: float = 2.0

    def __init__(self) -> None:
        self.source = "osm"
        self.name = "osm_beacons"
        self.priority = 30
        self.kinds = ("navaid",)
        self._data: None | pd.DataFrame = None

    @property
    def cache_file(self) -> Path:
        return cache_path / "traffic_osm_beacons.parquet"

    @staticmethod
    def _normalise_type(
        beacon_type: str,
        localizer: bool,
        glideslope: bool,
    ) -> str:
        """Map a raw OSM ``beacon:type`` to the traffic navaid taxonomy."""
        t = (beacon_type or "").strip().upper()
        if t == "NDB":
            return "NDB"
        if t in ("VOR", "DVOR"):
            return "VOR"
        if t in ("DVOR/DME", "DME"):
            return "DME"
        if t == "OM":
            return "OM"
        if t == "MM":
            return "MM"
        if t == "IM":
            return "IM"
        if t == "ILS":
            if glideslope:
                return "GS"
            if localizer:
                return "LOC"
            return "ILS"
        return t

    @staticmethod
    def _beacons_dataframe(overpass: Any) -> pd.DataFrame:
        """Turn a cartes ``Overpass`` into a normalised navaid DataFrame."""
        data = overpass.data
        columns = {
            "beacon:code": "beacon_code",
            "beacon:type": "beacon_type",
            "beacon:frequency": "frequency",
            "name": "name",
            "latitude": "latitude",
            "longitude": "longitude",
        }
        present = {
            src: dst for src, dst in columns.items() if src in data.columns
        }
        df = data[list(present.keys())].rename(columns=present).copy()

        if "beacon_type" not in df.columns:
            df["beacon_type"] = ""
        if "beacon_code" not in df.columns:
            df["beacon_code"] = None
        if "frequency" not in df.columns:
            df["frequency"] = None
        if "name" not in df.columns:
            df["name"] = None

        localizer = (
            data["localizer"].eq("yes")
            if "localizer" in data.columns
            else pd.Series(False, index=data.index)
        )
        glideslope = (
            data["glideslope"].eq("yes")
            if "glideslope" in data.columns
            else pd.Series(False, index=data.index)
        )
        df["type"] = [
            OSMBeaconsProvider._normalise_type(bt, loc, gs)
            for bt, loc, gs in zip(df["beacon_type"], localizer, glideslope)
        ]
        # code falls back to name then to ''
        df["name_code"] = df["beacon_code"].fillna(df["name"])
        df["altitude"] = 0.0
        return df

    def fetch_data(self) -> pd.DataFrame:
        import cartes.osm

        osm_beacons = getattr(cartes.osm, "beacons")
        overpass = osm_beacons()  # global pull (opt-in); cached by cartes
        if overpass.data.shape[0] == 0:
            return pd.DataFrame(
                columns=[
                    "name",
                    "type",
                    "latitude",
                    "longitude",
                    "altitude",
                    "frequency",
                    "description",
                ]
            )
        df = self._beacons_dataframe(overpass)
        out = pd.DataFrame(
            {
                "name": df["name_code"].fillna("").str.upper(),
                "type": df["type"],
                "latitude": df["latitude"].astype(float),
                "longitude": df["longitude"].astype(float),
                "altitude": 0.0,
                "frequency": pd.to_numeric(df["frequency"], errors="coerce"),
                "description": df["name"],
            }
        )
        return out

    def _refresh_cache(self) -> None:
        data = self.fetch_data()
        data.to_parquet(self.cache_file)

    @property
    def data(self) -> pd.DataFrame:
        """Bulk, globally-fetched OSM beacons (cached). Opt-in global pull."""
        if self._data is not None:
            return self._data

        if not self.cache_file.exists():
            self._refresh_cache()

        last_modification = self.cache_file.lstat().st_mtime
        delta = pd.Timestamp("now") - pd.Timestamp(last_modification * 1e9)
        if cache_expiration is not None and delta > cache_expiration:
            try:
                self._refresh_cache()
            except (httpx.TransportError, OSError):
                pass

        self._data = pd.read_parquet(self.cache_file)
        return self._data

    def resolve(
        self,
        code: str,
        *,
        kind: None | EntityKind = None,
        reference: None | tuple[float, float] = None,
        when: None | pd.Timestamp = None,
    ) -> list[ResolutionCandidate]:
        del when

        if kind not in (None, "navaid"):
            return []

        # Lazy/silent: no scope, no fetch. Keeps the default source cheap.
        if reference is None:
            return []

        try:
            import cartes.osm

            osm_beacons = getattr(cartes.osm, "beacons")
        except (ImportError, AttributeError):
            return []

        lat, lon = reference
        d = self.reference_half_extent
        # west, south, east, north
        bounds = (lon - d, lat - d, lon + d, lat + d)
        try:
            overpass = osm_beacons(bounds=bounds)  # cached by cartes
        except (httpx.TransportError, OSError, RuntimeError):
            return []  # Overpass errors must never break resolution.

        if overpass.data.shape[0] == 0:
            return []
        df = self._beacons_dataframe(overpass)

        upper = code.upper()
        codes = df["name_code"].fillna("").str.upper()
        names = df["name"].fillna("").str.upper()
        mask = (codes == upper) | (names == upper)
        subset = df[mask]

        out: list[ResolutionCandidate] = []
        for _, row in subset.iterrows():
            out.append(
                ResolutionCandidate(
                    code=str(row["name_code"]) or upper,
                    kind="navaid",
                    source=self.source,
                    confidence=0.5,
                    payload={
                        "name": str(row["name_code"] or ""),
                        "type": str(row["type"]),
                        "point_type": str(row["beacon_type"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude": 0.0,
                        "frequency": (
                            float(row["frequency"])
                            if pd.notna(row["frequency"])
                            else None
                        ),
                        "description": (
                            str(row["name"]) if pd.notna(row["name"]) else None
                        ),
                    },
                )
            )
        return out
