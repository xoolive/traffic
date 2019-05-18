from traffic.data import eurofirs, navaids


def test_getter() -> None:
    narak = navaids["NARAK"]
    assert narak is not None
    assert narak.latlon == (44.295278, 1.748889)
    assert narak.type == "FIX"


def test_extent() -> None:
    gaithersburg = navaids["GAI"]
    assert gaithersburg is not None
    assert gaithersburg.type == "NDB"
    gaillac = navaids.extent(eurofirs["LFBB"])["GAI"]
    assert gaillac is not None
    assert gaillac.type == "VOR"
    assert gaillac.latlon == (43.954056, 1.824167)


def test_search() -> None:
    assert navaids.search("GAI").data.shape[0] == 2
