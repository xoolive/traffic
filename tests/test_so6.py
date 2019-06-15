from datetime import timedelta
from pathlib import Path

from traffic.data import SO6, eurofirs


def test_so6():
    so6_path = Path(__file__).parent.parent / "data" / "sample_m3.so6.7z"
    so6 = SO6.from_file(so6_path)

    assert so6 is not None
    assert len(so6) == 11043

    hop36pp = so6["HOP36PP"]

    assert hop36pp.origin == "LFML"
    assert hop36pp.destination == "LFBD"
    assert hop36pp.aircraft == "A319"
    assert hop36pp.flight_id == 332206265
    assert so6[332206265].callsign == "HOP36PP"

    assert hop36pp.intersects(eurofirs["LFBB"])
    assert not hop36pp.intersects(eurofirs["LFEE"])

    clipped = so6["HOP36PP"].clip(eurofirs["LFBB"].flatten())
    assert clipped is not None
    assert str(clipped.start)[:-13] == "2018-01-01 18:30:20"

    assert 26638 < hop36pp.at("2018/01/01 18:40").altitude < 26639
    assert hop36pp.between(
        "2018/01/01 18:25", timedelta(minutes=30)
    ).intersects(eurofirs["LFBB"])
    assert not hop36pp.between(
        "2018/01/01 18:25", timedelta(minutes=5)
    ).intersects(eurofirs["LFBB"])

    noon = so6.at("2018/01/01 12:00")
    bdx_flights = noon.inside_bbox(eurofirs["LFBB"]).intersects(
        eurofirs["LFBB"]
    )
    assert len(bdx_flights) == 3
