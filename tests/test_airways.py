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

    swiss_length = max(
        a.project_shape().length
        for a in airways.extent(eurofirs["LSAS"]).through("DITON")
    )
    full_length = max(
        a.project_shape().length for a in airways.through("DITON")
    )
    assert swiss_length < 1e6 < full_length

    short_un871 = airways.extent(eurofirs["LFBB"])["UN871"]
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
