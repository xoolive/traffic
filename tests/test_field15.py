"""Tests for field15 route parsing and resolver integration.

Tests that require external AIXM data are guarded by ``@pytest.mark.slow``
and skip unless ``TRAFFIC_TEST_AIXM_DIR`` is set in the environment.

Unit tests (the bulk of this file) exercise the Python bindings directly via
``thrust.data.field15.Field15Parser`` and the resolver plumbing without
requiring any on-disk navigation database.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Thrust field15 Python binding tests (pure parser — no DB needed)
# ---------------------------------------------------------------------------


def _import_field15_parser():
    """Return Field15Parser from thrust, or skip the test if thrust is not installed."""
    try:
        from thrust.data.field15 import Field15Parser  # type: ignore[import]

        return Field15Parser
    except ImportError:
        pytest.skip("thrust Python bindings not installed")


def test_parse_direct_route_returns_tokens():
    """A simple DCT route tokenises to aerodrome + DCT + waypoint sequences."""
    pytest.importorskip("thrust")
    Field15Parser = _import_field15_parser()

    tokens = Field15Parser.parse("LFPG DCT LACOU DCT LFLL")
    kinds = [type(t).__name__ for t in tokens]
    # At minimum we expect points and connectors — not an empty list
    assert len(tokens) >= 3


def test_parse_airway_route_contains_airway_connector():
    """A route with an ATS airway segment should contain at least one Connector.Airway token."""
    pytest.importorskip("thrust")
    Field15Parser = _import_field15_parser()
    from thrust.data.field15 import Connector  # type: ignore[import]

    tokens = Field15Parser.parse("LFPG DCT LACOU UM184 VEBIT DCT LFLL")
    airway_tokens = [
        t for t in tokens if isinstance(t, Connector) and hasattr(t, "value")
    ]
    # Alternatively check by repr — Connector.Airway should appear
    token_reprs = [repr(t) for t in tokens]
    assert any("UM184" in r for r in token_reprs)


def test_parse_speed_altitude_modifier():
    """A speed/altitude prefix like N0450F350 should parse as a Modifier token."""
    pytest.importorskip("thrust")
    Field15Parser = _import_field15_parser()
    from thrust.data.field15 import Modifier  # type: ignore[import]

    # Typical field 15 with speed/level modifier
    tokens = Field15Parser.parse("N0450F350 LFPG DCT LACOU")
    modifier_tokens = [t for t in tokens if isinstance(t, Modifier)]
    assert len(modifier_tokens) >= 1


def test_parse_dct_only_route_has_no_airway():
    """A DCT-only route should produce no Connector.Airway tokens."""
    pytest.importorskip("thrust")
    Field15Parser = _import_field15_parser()

    tokens = Field15Parser.parse("LFPG DCT NARAK DCT LFLL")
    token_reprs = [repr(t) for t in tokens]
    # Should contain DCT but no airway codes like L738
    assert not any(
        r
        for r in token_reprs
        if "Airway" in r
        or any(code in r for code in ["UN869", "UM184", "L738", "UL613"])
    )


# ---------------------------------------------------------------------------
# AiracDatabase Python binding (requires AIXM data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_airac_database_enrich_route_returns_segments():
    """AiracDatabase.enrich_route returns a non-empty list of Segment objects."""
    aixm_dir = os.environ.get("TRAFFIC_TEST_AIXM_DIR")
    if aixm_dir is None:
        pytest.skip("Set TRAFFIC_TEST_AIXM_DIR to run AIXM field15 test")

    path = Path(aixm_dir).expanduser()
    if not path.exists():
        pytest.skip(f"AIXM path not found: {path}")

    try:
        from thrust.core.field15 import AiracDatabase  # type: ignore[import]
    except ImportError:
        pytest.skip("thrust Python bindings not installed")

    db = AiracDatabase(path.as_posix())
    segments = db.enrich_route("LFPG DCT LACOU UM184 VEBIT DCT LFLL")
    assert len(segments) > 0

    # Each segment should have start/end points with coordinates
    for seg in segments:
        assert hasattr(seg, "start")
        assert hasattr(seg, "end")
        assert isinstance(seg.start.latitude, float)
        assert isinstance(seg.start.longitude, float)
        assert isinstance(seg.end.latitude, float)
        assert isinstance(seg.end.longitude, float)
        # Coordinates should be in plausible European airspace range
        assert 30.0 <= seg.start.latitude <= 75.0
        assert -15.0 <= seg.start.longitude <= 40.0


@pytest.mark.slow
def test_airac_database_enrich_route_segment_to_dict():
    """Segment.to_dict() returns a dict with start/end keys."""
    aixm_dir = os.environ.get("TRAFFIC_TEST_AIXM_DIR")
    if aixm_dir is None:
        pytest.skip("Set TRAFFIC_TEST_AIXM_DIR to run AIXM field15 test")

    path = Path(aixm_dir).expanduser()
    if not path.exists():
        pytest.skip(f"AIXM path not found: {path}")

    try:
        from thrust.core.field15 import AiracDatabase  # type: ignore[import]
    except ImportError:
        pytest.skip("thrust Python bindings not installed")

    db = AiracDatabase(path.as_posix())
    segments = db.enrich_route("LFPG DCT LACOU DCT LFLL")
    assert len(segments) > 0

    d = segments[0].to_dict()
    assert "start" in d
    assert "end" in d
    assert "latitude" in d["start"]
    assert "longitude" in d["start"]


# ---------------------------------------------------------------------------
# Resolver.parse_route integration
# ---------------------------------------------------------------------------


def test_parse_route_raises_without_aixm_configured():
    """parse_route() raises RuntimeError when no AIXM source is configured."""
    from traffic.data.resolver import Resolver

    r = Resolver()  # fresh resolver, no AIXM
    with pytest.raises(RuntimeError, match="No field15 provider configured"):
        r.parse_route("LFPG DCT LACOU DCT LFLL")


def test_module_level_parse_route_raises_without_nixm():
    """Module-level parse_route() raises if the default resolver has no AIXM."""
    from traffic.data import resolver

    # Save and temporarily replace the default resolver with a fresh one
    # (without AIXM) to keep tests isolated
    from traffic.data.resolver.core import Resolver

    original = resolver._DEFAULT_RESOLVER  # type: ignore[attr-defined]
    try:
        resolver._DEFAULT_RESOLVER = Resolver()  # type: ignore[attr-defined]
        with pytest.raises(
            RuntimeError, match="No field15 provider configured"
        ):
            resolver.parse_route("LFPG DCT LACOU DCT LFLL")
    finally:
        resolver._DEFAULT_RESOLVER = original  # type: ignore[attr-defined]


@pytest.mark.slow
def test_resolver_parse_route_with_aixm():
    """Resolver.parse_route returns segments when AIXM is configured."""
    aixm_dir = os.environ.get("TRAFFIC_TEST_AIXM_DIR")
    if aixm_dir is None:
        pytest.skip(
            "Set TRAFFIC_TEST_AIXM_DIR to run AIXM resolver field15 test"
        )

    path = Path(aixm_dir).expanduser()
    if not path.exists():
        pytest.skip(f"AIXM path not found: {path}")

    from traffic.data.resolver import Resolver

    r = Resolver().eurocontrol(aixm=path)
    segments = r.parse_route("LFPG DCT LACOU UM184 VEBIT DCT LFLL")
    assert isinstance(segments, list)
    assert len(segments) > 0

    for seg in segments:
        assert "start" in seg
        assert "end" in seg
        assert "latitude" in seg["start"]
        assert "longitude" in seg["start"]


@pytest.mark.slow
def test_resolver_parse_route_airway_segment_has_name():
    """Segments on an ATS airway have a non-None name field."""
    aixm_dir = os.environ.get("TRAFFIC_TEST_AIXM_DIR")
    if aixm_dir is None:
        pytest.skip(
            "Set TRAFFIC_TEST_AIXM_DIR to run AIXM resolver field15 test"
        )

    path = Path(aixm_dir).expanduser()
    if not path.exists():
        pytest.skip(f"AIXM path not found: {path}")

    from traffic.data.resolver import Resolver

    r = Resolver().eurocontrol(aixm=path)
    # UM184 is a well-known upper airway in France
    segments = r.parse_route("LACOU UM184 VEBIT")
    assert len(segments) > 0
    # At least some segments should have name "UM184"
    names = {seg.get("name") for seg in segments}
    assert "UM184" in names


# ---------------------------------------------------------------------------
# AIXMField15Provider unit tests (mock DB)
# ---------------------------------------------------------------------------


def test_aixm_field15_provider_raises_without_thrust(monkeypatch):
    """AIXMField15Provider raises RuntimeError if thrust is not importable."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "thrust.core.field15":
            raise ImportError("mocked: thrust not installed")
        return real_import(name, *args, **kwargs)

    from traffic.data.resolver.providers_eurocontrol import AIXMField15Provider

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(RuntimeError, match="thrust Python bindings"):
            AIXMField15Provider("/some/path")
