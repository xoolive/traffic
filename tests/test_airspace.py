import pytest

from traffic.core import Airspace
from traffic.data import eurofirs

try:
    from traffic.data import nm_airspaces
except RuntimeError:
    nm_data = False
else:
    nm_data = True


def test_airspace() -> None:
    EBBU = eurofirs["EBBU"]
    EHAA = eurofirs["EHAA"]
    assert EBBU is not None
    assert EHAA is not None

    summed = EBBU + EHAA
    lon1, lon2, lat1, lat2 = summed.extent
    lat, lon = summed.point.latlon

    assert lat1 < lat < lat2
    assert lon1 < lon < lon2
    assert summed.area > EBBU.area
    assert summed.area > EHAA.area
    assert summed.above(300).area == EHAA.area
    assert (summed.below(190).area - summed.area) / summed.area < 1e-12

    json = summed.export_json()
    assert json["name"] == "BRUSSELS FIR, AMSTERDAM FIR"
    assert Airspace.from_json(json).area == summed.area


def test_area() -> None:
    for fir in eurofirs:
        assert fir.area > 1e6


@pytest.mark.skipif(not nm_data, reason="No NM data available")
def test_nm() -> None:
    maastricht = nm_airspaces["EDYYUTAX"]
    assert maastricht is not None
    assert maastricht.area > 1e11
