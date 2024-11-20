import pytest

from traffic.core import Flight
from traffic.data.adsb.decode import RawData
from traffic.data.samples import fr24, sample_dump1090, switzerland


def long_enough(flight: Flight) -> bool:
    return len(flight) > 100


def test_simple() -> None:
    f = fr24.flight("34a8254b")
    g = f.first("25 min").query_opensky()
    assert g is not None
    h = g.query_ehs()
    assert h.data.shape[0] > g.data.shape[0]


@pytest.mark.skipif(True, reason="only for local debug")
def test_decode() -> None:
    tap_switzerland = (
        switzerland.query(  # type: ignore
            'callsign.str.startswith("TAP127")', engine="python"
        )
        .iterate_lazy()
        .pipe(long_enough)
        .query_opensky()
        .resample("1s")
        .query_ehs(progressbar=False)
        .filter(selected_mcp=23)
        .filter(altitude=53, selected_mcp=53, roll=53, heading=53)
        .resample("1s")
        .eval()
    )

    # BDS 4,0
    for f in tap_switzerland:
        # This is only safe en route. Even if the selected MCP altitude
        # changes, altitude should be between min/max (modulo data errors)
        assert all(
            (f.min("selected_mcp") - 100 <= f.data.altitude)
            & (f.data.altitude <= f.max("selected_mcp") + 100)
        )

    # BDS 5,0 and BDS 6,0
    for f in tap_switzerland:
        # An aircraft should turn to the side it is rolling
        f = f.assign(diff_heading=lambda df: df.heading.diff() * df.roll)
        assert sum(f.data.diff_heading + 1 < 0) / len(f) < 1e-3


def test_dump1090_bin() -> None:
    time_0 = "2020-02-12 10:07Z"
    t = RawData.from_dump1090_output(
        sample_dump1090, "LFBO", reference_time=time_0
    )

    assert t is not None
    assert t.data.shape[0] != 0
