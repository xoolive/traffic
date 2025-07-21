import pickle

from traffic.data import eurofirs, navaids


def test_getter() -> None:
    narak = navaids["NARAK"]
    assert narak.latlon == (44.29527778, 1.74888889)
    assert narak.lat == 44.29527778
    assert narak.lon == 1.74888889
    assert narak.type == "FIX"


def test_extent() -> None:
    gaithersburg = navaids["GAI"]
    assert gaithersburg.type == "NDB"
    LFBB = eurofirs["LFBB"]
    assert LFBB is not None
    nav_ext = navaids.extent(LFBB)
    assert nav_ext is not None
    gaillac = nav_ext["GAI"]
    assert gaillac.type == "VOR"
    assert gaillac.latlon == (43.95405556, 1.82416667)


def test_search() -> None:
    assert navaids.search("GAI").data.shape[0] == 2


def test_iter() -> None:
    extent = (1, 2, 43, 44)
    nav_ext = navaids.extent(extent)
    assert nav_ext is not None
    assert sum(1 for n in nav_ext if n.name == "GAI") == 1


def test_pickling() -> None:
    original = navaids["GAI"]
    p = pickle.dumps(original)
    restored = pickle.loads(p)
    assert restored is not None
