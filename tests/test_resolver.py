from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import pandas as pd
from traffic.data import (
    airports,
    airspaces,
    airways,
    fixes,
    freeroute,
    navaids,
    resolver,
)
from traffic.data.eurocontrol.ddr.airspaces import NMAirspaceParser
from traffic.data.resolver.models import ResolutionCandidate


def test_sources_frame_has_default_airport_sources() -> None:
    frame = resolver.sources_frame()
    labels = set(frame["label"])
    assert {"flightradar24", "ourairports"}.issubset(labels)


@pytest.mark.parametrize("code", ["LFBO", "LFCL", "LFCX"])
def test_airports_getitem_resolves_common_codes(code: str) -> None:
    airport = airports[code]
    assert airport is not None
    assert airport.icao == code


def test_resolver_can_force_ourairports_source() -> None:
    result = resolver.resolve(airport="LFCX", source="ourairports")
    assert result.selected is not None
    assert result.selected.source == "ourairports"
    assert result.selected.code == "LFCX"


def test_resolver_default_priority_prefers_fr24(
    requires_network: None,
) -> None:
    result = resolver.resolve(airport="LFPG")
    assert result.selected is not None
    assert result.selected.source == "flightradar24"


def test_airways_getitem_uses_resolver() -> None:
    route = airways["UN869"]
    assert route is not None
    assert route.name == "UN869"
    assert len(route.navaids) > 2


def test_source_shortcuts_equivalence() -> None:
    assert (
        airports.source("ourairports")["LFCX"].icao
        == airports.get("LFCX", source="ourairports").icao
    )
    assert (
        airways.source("xplane")["UN869"].name
        == airways.get("UN869", source="xplane").name
    )
    assert (
        fixes.source("xplane")["NARAK"].name
        == fixes.get("NARAK", source="xplane").name
    )


def test_navaid_fix_query_semantics_are_explicit() -> None:
    assert navaids.get_fix("NARAK").is_fix
    assert not navaids.get_navaid("GAI").is_fix


def test_resolver_resolves_airway_from_default_source() -> None:
    result = resolver.resolve(airway="UN869", source="xplane")
    assert result.selected is not None
    assert result.selected.source == "xplane"
    assert result.selected.kind == "airway"


@pytest.mark.slow
def test_activate_nasr_source_if_available() -> None:
    nasr_path = os.environ.get("TRAFFIC_TEST_NASR_ZIP")
    if nasr_path is None:
        pytest.skip("Set TRAFFIC_TEST_NASR_ZIP to run NASR resolver test")

    path = Path(nasr_path).expanduser()
    if not path.exists():
        pytest.skip(f"NASR file not found: {path}")

    resolver.faa(nasr=path)
    labels = set(resolver.sources_frame()["label"])
    assert "faa_nasr" in labels

    result = resolver.resolve(airport="KLAX", source="faa_nasr")
    assert result.selected is not None
    assert result.selected.source == "faa_nasr"

    navaid = resolver.resolve(navaid="LAX", source="faa_nasr")
    assert navaid.selected is not None
    assert navaid.selected.source == "faa_nasr"
    assert navaid.selected.kind == "navaid"

    # KLAX airport coordinates (NASR 2026-03-19 ground truth)
    klax = resolver.resolve(airport="KLAX", source="faa_nasr")
    assert klax.selected is not None
    klax_payload = klax.selected.payload
    assert abs(klax_payload["latitude"] - 33.94249638) < 0.0001, (
        f"KLAX lat off: {klax_payload['latitude']}"
    )
    assert abs(klax_payload["longitude"] - (-118.40804861)) < 0.0001, (
        f"KLAX lon off: {klax_payload['longitude']}"
    )

    # BAF VORTAC: identifier, frequency, and coordinates.
    # The Python Navaid.name attribute holds the identifier code ("BAF"), not
    # the station name ("BARNES").  The station name lives in .description and
    # payload["name"].  BAF has no separate fix record in NASR — resolve_fix
    # falls
    # back to the VORTAC navaid record.
    baf = navaids.source("faa_nasr").get("BAF")
    assert baf.name == "BAF"  # identifier
    assert "BARNES" in (baf.description or ""), (
        f"BAF description should contain BARNES: {baf.description}"
    )
    assert baf.frequency is not None
    assert float(baf.frequency) > 100
    assert abs(float(baf.latitude) - 42.16195908) < 0.0001, (
        f"BAF lat off: {baf.latitude}"
    )
    assert abs(float(baf.longitude) - (-72.7161995)) < 0.0001, (
        f"BAF lon off: {baf.longitude}"
    )

    # BASYE fix coordinates (NASR 2026-03-19 ground truth)
    basye_result = resolver.resolve(fix="BASYE", source="faa_nasr")
    assert basye_result.selected is not None
    basye_payload = basye_result.selected.payload
    assert abs(basye_payload["latitude"] - 41.34372222) < 0.0001, (
        f"BASYE lat off: {basye_payload['latitude']}"
    )
    assert abs(basye_payload["longitude"] - (-73.79860833)) < 0.0001, (
        f"BASYE lon off: {basye_payload['longitude']}"
    )

    # Q448 waypoint sequence and BASYE coordinate within the airway definition
    q448 = resolver.resolve(airway="Q448", source="faa_nasr")
    assert q448.selected is not None
    q448_points = q448.selected.payload["points"]
    q448_codes = [x["code"] for x in q448_points]
    assert q448_codes == [
        "PTW",
        "LANNA",
        "DBABE",
        "BASYE",
        "TRIBS",
        "BIGGO",
        "BAF",
    ]
    # BASYE is the 4th point (index 3) — verify its coordinates match the
    # fix resolution
    basye_in_q448 = q448_points[3]
    assert basye_in_q448["code"] == "BASYE"
    assert abs(basye_in_q448.get("latitude", 0.0) - 41.34372222) < 0.0001, (
        f"BASYE lat in Q448 off: {basye_in_q448.get('latitude')}"
    )
    assert abs(basye_in_q448.get("longitude", 0.0) - (-73.79860833)) < 0.0001, (
        f"BASYE lon in Q448 off: {basye_in_q448.get('longitude')}"
    )

    # J48 waypoint sequence and LANNA coordinate (NASR 2026-03-19 ground truth)
    j48 = resolver.resolve(airway="J48", source="faa_nasr")
    assert j48.selected is not None
    j48_points = j48.selected.payload["points"]
    j48_codes = [x["code"] for x in j48_points]
    assert j48_codes == [
        "LANNA",
        "PTW",
        "BYRDD",
        "HAAGN",
        "PENSY",
        "EMI",
        "CSN",
        "MOL",
    ], f"J48 sequence off: {j48_codes}"
    # LANNA is the first point — verify its coordinates
    lanna = j48_points[0]
    assert abs(lanna.get("latitude", 0.0) - 40.55974166) < 0.0001, (
        f"LANNA lat off: {lanna.get('latitude')}"
    )
    assert abs(lanna.get("longitude", 0.0) - (-75.027725)) < 0.0001, (
        f"LANNA lon off: {lanna.get('longitude')}"
    )
    assert abs(float(baf.longitude) - (-72.7161995)) < 0.0001, (
        f"BAF lon off: {baf.longitude}"
    )

    # BASYE fix coordinates (NASR 2026-03-19 ground truth)
    basye = resolver.resolve(fix="BASYE", source="faa_nasr")
    assert basye.selected is not None
    basye_lat = basye.selected.payload.get("latitude", 0.0)
    basye_lon = basye.selected.payload.get("longitude", 0.0)
    assert abs(basye_lat - 41.34372222) < 0.0001, f"BASYE lat off: {basye_lat}"
    assert abs(basye_lon - (-73.79860833)) < 0.0001, (
        f"BASYE lon off: {basye_lon}"
    )

    # Q448 waypoint sequence and BASYE coordinate within the airway definition
    q448 = resolver.resolve(airway="Q448", source="faa_nasr")
    assert q448.selected is not None
    q448_points = q448.selected.payload["points"]
    q448_codes = [x["code"] for x in q448_points]
    assert q448_codes == [
        "PTW",
        "LANNA",
        "DBABE",
        "BASYE",
        "TRIBS",
        "BIGGO",
        "BAF",
    ]
    # BASYE is the 4th point (index 3) — verify its coordinates are present
    # and correct
    basye_in_q448 = q448_points[3]
    assert basye_in_q448["code"] == "BASYE"
    assert abs(basye_in_q448.get("latitude", 0.0) - 41.34372222) < 0.0001, (
        f"BASYE lat in Q448 off: {basye_in_q448.get('latitude')}"
    )
    assert abs(basye_in_q448.get("longitude", 0.0) - (-73.79860833)) < 0.0001, (
        f"BASYE lon in Q448 off: {basye_in_q448.get('longitude')}"
    )

    # J48 waypoint sequence and LANNA coordinate
    j48 = resolver.resolve(airway="J48", source="faa_nasr")
    assert j48.selected is not None
    j48_points = j48.selected.payload["points"]
    j48_codes = [x["code"] for x in j48_points]
    assert j48_codes == [
        "LANNA",
        "PTW",
        "BYRDD",
        "HAAGN",
        "PENSY",
        "EMI",
        "CSN",
        "MOL",
    ], f"J48 sequence off: {j48_codes}"
    # LANNA is the first point — verify its coordinates
    lanna = j48_points[0]
    assert abs(lanna.get("latitude", 0.0) - 40.55974166) < 0.0001, (
        f"LANNA lat off: {lanna.get('latitude')}"
    )
    assert abs(lanna.get("longitude", 0.0) - (-75.027725)) < 0.0001, (
        f"LANNA lon off: {lanna.get('longitude')}"
    )

    faa_spaces = resolver.data(source="faa_nasr", kind="airspace")
    assert not faa_spaces.empty
    dsg = str(faa_spaces.iloc[0]["designator"])
    s = resolver.resolve(airspace=dsg, source="faa_nasr")
    assert s.selected is not None


@pytest.mark.slow
def test_activate_aixm_source_if_available() -> None:
    aixm_dir = os.environ.get("TRAFFIC_TEST_AIXM_DIR")
    if aixm_dir is None:
        pytest.skip("Set TRAFFIC_TEST_AIXM_DIR to run AIXM resolver test")

    path = Path(aixm_dir).expanduser()
    if not path.exists():
        pytest.skip(f"AIXM path not found: {path}")

    resolver.eurocontrol(aixm=path)
    labels = set(resolver.sources_frame()["label"])
    assert "eurocontrol_aixm" in labels

    result = resolver.resolve(airport="LFCX", source="eurocontrol_aixm")
    if result.selected is None:
        pytest.skip("LFCX not present in provided AIXM cycle")

    assert result.selected.source == "eurocontrol_aixm"

    spaces = airspaces.source("eurocontrol_aixm").data
    if spaces.empty:
        pytest.skip("No airspaces found in provided AIXM cycle")

    designator = str(spaces.iloc[0]["designator"])
    space = resolver.resolve(airspace=designator, source="eurocontrol_aixm")
    assert space.selected is not None


def test_resolve_fix_narak_default_source() -> None:
    result = resolver.resolve(fix="NARAK")
    assert result.selected is not None
    assert result.selected.source == "xplane"
    assert result.selected.kind == "fix"


def test_airspaces_type_and_search_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = pd.DataFrame(
        [
            {
                "designator": "LFBBCTA",
                "name": "BORDEAUX FRA",
                "type": "FRA",
            },
            {
                "designator": "LFFFCTA",
                "name": "PARIS CTA",
                "type": "CTA",
            },
        ]
    )

    monkeypatch.setattr(resolver, "data", lambda **_: sample)

    filtered = airspaces.type("fra").search("LFBB").data
    assert list(filtered["designator"]) == ["LFBBCTA"]


def test_freeroute_alias_matches_airspaces_fra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = pd.DataFrame(
        [
            {"designator": "LFFRA", "name": "France FRA", "type": "FRA"},
            {"designator": "LFCTA", "name": "France CTA", "type": "CTA"},
        ]
    )

    monkeypatch.setattr(resolver, "data", lambda **_: sample)

    assert list(freeroute.data["designator"]) == list(
        airspaces.fra.data["designator"]
    )


def test_airspaces_get_handles_layered_payload_with_shapely_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SimpleNamespace(
        code="LFBBCTA",
        payload={
            "designator": "LFBBCTA",
            "name": "BORDEAUX U/ACC",
            "type": "AUA",
            "layers": [
                {
                    "lower": 195.0,
                    "upper": 295.0,
                    "coordinates": [
                        (1.0, 44.0),
                        (2.0, 44.0),
                        (2.0, 45.0),
                        (1.0, 45.0),
                    ],
                },
                {
                    "lower": 195.0,
                    "upper": 295.0,
                    "coordinates": [
                        (1.5, 44.5),
                        (2.5, 44.5),
                        (2.5, 45.5),
                        (1.5, 45.5),
                    ],
                },
            ],
        },
    )
    result = SimpleNamespace(selected=candidate, alternatives=[])

    monkeypatch.setattr(resolver, "resolve", lambda **_: result)

    space = airspaces.get("LFBBCTA", source="eurocontrol_ddr")
    assert space.designator == "LFBBCTA"
    assert len(space.elements) == 1


@pytest.mark.slow
def test_airspaces_get_matches_nm_airspaces_for_lfbb_designators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nm_dir = os.environ.get("TRAFFIC_TEST_NM_DIR")
    if nm_dir is None:
        pytest.skip("Set TRAFFIC_TEST_NM_DIR to run NM airspace parity test")

    nm_path = Path(nm_dir).expanduser()
    if not nm_path.exists():
        pytest.skip(f"NM path not found: {nm_path}")

    NMAirspaceParser.nm_path = nm_path
    nm = NMAirspaceParser(data=None, config_file=None)

    targets = ["LFBBBDX", "LFBBRL", "LFBBR1"]
    payload_by_code: dict[str, dict[str, object]] = {}
    expected_by_code: dict[str, list[tuple[float | None, float | None]]] = {}

    for code in targets:
        expected = nm[code]
        expected_by_code[code] = sorted(
            [(elt.lower, elt.upper) for elt in expected.elements],
            key=lambda item: (
                float("inf") if item[0] is None else float(item[0]),
                float("inf") if item[1] is None else float(item[1]),
            ),
        )

        rows = (
            nm.consolidate()
            .data.query("designator == @code")
            .dropna(subset=["geometry", "lower", "upper"])
        )
        if rows.empty:
            pytest.skip(f"No consolidated rows found for {code}")

        layers = [
            {
                "lower": float(row.lower),
                "upper": float(row.upper),
                "coordinates": [
                    (float(lon), float(lat))
                    for (lon, lat, *_) in row.geometry.exterior.coords
                ],
            }
            for row in rows.itertuples()
        ]
        payload_by_code[code] = {
            "designator": code,
            "name": str(rows.iloc[0].get("name") or code),
            "type": str(rows.iloc[0].get("type") or ""),
            "layers": layers,
        }

    def fake_resolve(*, airspace: str | None = None, **_: object) -> object:
        assert airspace is not None
        payload = payload_by_code[airspace]
        selected = SimpleNamespace(code=airspace, payload=payload)
        return SimpleNamespace(selected=selected, alternatives=[])

    monkeypatch.setattr(resolver, "resolve", fake_resolve)

    for code in targets:
        got = airspaces.get(code, source="eurocontrol_ddr")
        bands = sorted(
            [(elt.lower, elt.upper) for elt in got.elements],
            key=lambda item: (
                float("inf") if item[0] is None else float(item[0]),
                float("inf") if item[1] is None else float(item[1]),
            ),
        )
        assert bands == expected_by_code[code]


def test_resolver_falls_through_provider_errors() -> None:
    class BrokenProvider:
        source = "broken"
        name = "broken"
        priority = 100

        def resolve(self, code: str, **_: object) -> list[ResolutionCandidate]:
            raise RuntimeError("provider unavailable")

    class WorkingProvider:
        source = "working"
        name = "working"
        priority = 1

        def resolve(self, code: str, **_: object) -> list[ResolutionCandidate]:
            return [
                ResolutionCandidate(
                    code=code,
                    kind="airport",
                    source=self.source,
                    confidence=1.0,
                )
            ]

    result = resolver.Resolver([BrokenProvider(), WorkingProvider()]).resolve(
        airport="TEST"
    )
    assert result.selected is not None
    assert result.selected.source == "working"
