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
        a.name for a in airways.through("NARAK") if a.name.startswith("U")
    )
    assert narak_airways == {"UN859", "UN869", "UT122", "UY155", "UZ365"}

    air_ext = airways.extent(eurofirs["LSAS"])
    assert air_ext is not None
    swiss_length = max(
        a.project_shape().length for a in air_ext.through("DITON")
    )
    full_length = max(
        a.project_shape().length for a in airways.through("DITON")
    )
    assert swiss_length < 1e6 < full_length

    air_ext = airways.extent(eurofirs["LFBB"])
    assert air_ext is not None

    short_un871 = air_ext["UN871"]
    assert short_un871 is not None
    assert short_un871.navaids == [
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
