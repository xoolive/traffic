import pytest

import pandas as pd

try:
    from traffic.data import nm_b2b
except ImportError:
    no_import = True
else:
    no_import = False


today = pd.Timestamp("now")


@pytest.mark.skipif(no_import, reason="No key available")
def test_success() -> None:
    res = nm_b2b.flight_list(today, origin="LFBO")
    assert res is not None
    assert all(res.data.origin == "LFBO")


@pytest.mark.skipif(no_import, reason="No key available")
def test_error() -> None:
    with pytest.raises(RuntimeError):
        nm_b2b.flight_list(today, origin="LBO")
