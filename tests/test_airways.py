import pytest

from traffic.data import airways, eurofirs


def test_basic() -> None:
    foo = airways["FOO"]
    assert foo is None

    l888 = airways["L888"]
    assert l888 is not None
    assert 2.8e6 < l888.project_shape().length < 3e6


def test_through_extent() -> None:
    narak_airways = set(
        route
        for route in airways.search("NARAK").data.route
        if route.startswith("U")
    )
    assert narak_airways == {"UN859", "UN869", "UT122", "UY155", "UZ365"}

    LSAS = eurofirs["LSAS"]
    assert LSAS is not None
    air_ext = airways.extent(LSAS)
    assert air_ext is not None
    swiss_length = max(
        air_ext[route].project_shape().length  # type: ignore
        for route in air_ext.search("DITON").data.route
    )
    full_length = max(
        airways[route].project_shape().length  # type: ignore
        for route in airways.search("DITON").data.route
    )
    assert swiss_length < 1e6 < full_length

    LFBB = eurofirs["LFBB"]
    assert LFBB is not None
    air_ext = airways.extent(LFBB)
    assert air_ext is not None

    short_un871 = air_ext["UN871"]
    assert short_un871 is not None
    assert list(navaid.name for navaid in short_un871.navaids) == [
        "LARDA",
        "RONNY",
        "TOPTU",
        "GONUP",
        "TOU",
        "GAI",
        "MAKIL",
        "DITEV",
        "MEN",
    ]

    assert len(short_un871["LARDA", "TOPTU"].shape.coords) == 3
    assert len(short_un871["TOPTU", "LARDA"].shape.coords) == 3

    with pytest.raises(RuntimeError):
        short_un871["LARDA", "LARDA"]
    with pytest.raises(ValueError):
        short_un871["ERROR", "LARDA"]
    with pytest.raises(ValueError):
        short_un871["LARDA", "ERROR"]
