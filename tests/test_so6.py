from datetime import timedelta

import pytest
from traffic.data import eurofirs

notfound = False
try:
    from traffic.data.samples import sample_m3
except ModuleNotFoundError:
    notfound = True


@pytest.mark.skipif(notfound, reason="libarchive not available")
def test_so6() -> None:
    assert len(sample_m3) == 11043

    hop36pp = sample_m3["HOP36PP"]

    assert hop36pp.origin == "LFML"
    assert hop36pp.destination == "LFBD"
    assert hop36pp.typecode == "A319"
    assert hop36pp.flight_id == 332206265  # type: ignore
    assert sample_m3[332206265].callsign == "HOP36PP"

    LFBB = eurofirs["LFBB"]
    LFEE = eurofirs["LFEE"]

    assert LFBB is not None
    assert LFEE is not None

    assert hop36pp.intersects(LFBB)
    assert not hop36pp.intersects(LFEE)

    clipped = sample_m3["HOP36PP"].clip(LFBB.flatten())
    assert clipped is not None
    assert str(clipped.start)[:19] == "2018-01-01 18:30:20"
    assert str(clipped.stop)[:19] == "2018-01-01 18:52:10"

    assert len(next(hop36pp.clip_altitude(15000, 20000))) == 2
    assert len(next(hop36pp.clip_altitude(18000, 20000))) == 1

    # This flight is on holding pattern and causes issues with intersection
    clipped = sample_m3["BAW3TV"].clip(LFBB.flatten())
    assert clipped is not None
    assert str(clipped.start)[:19] == "2018-01-01 14:51:19"
    assert str(clipped.stop)[:19] == "2018-01-01 16:30:30"

    assert (
        sum(1 for _ in sample_m3["HOP36PP"].coords4d())
        == len(sample_m3["HOP36PP"]) + 1
    )

    assert 26638 < hop36pp.at("2018/01/01 18:40").altitude < 26639
    assert hop36pp.between(
        "2018/01/01 18:25", timedelta(minutes=30)
    ).intersects(LFBB)
    assert not hop36pp.between(
        "2018/01/01 18:25", timedelta(minutes=5)
    ).intersects(LFBB)

    noon = sample_m3.at("2018/01/01 12:00")
    bdx_flights = noon.inside_bbox(LFBB).intersects(LFBB)
    assert len(bdx_flights) == 3

    assert bdx_flights.data.shape[0] == 3
    select = sample_m3[bdx_flights]
    assert select is not None
    assert select.data.shape[0] == 28
