from traffic.core import Airspace
from traffic.data import eurofirs


def test_airspace() -> None:
    summed = eurofirs["EBBU"] + eurofirs["EHAA"]
    lon1, lon2, lat1, lat2 = summed.extent
    lat, lon = summed.point.latlon

    assert lat1 < lat < lat2
    assert lon1 < lon < lon2
    assert summed.area > eurofirs["EBBU"].area
    assert summed.area > eurofirs["EHAA"].area
    assert summed.above(300).area == eurofirs["EHAA"].area
    assert (summed.below(190).area - summed.area) / summed.area < 1e-12

    json = summed.export_json()
    assert json["name"] == "BRUSSELS, AMSTERDAM"
    assert Airspace.from_json(json).area == summed.area
