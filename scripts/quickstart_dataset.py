# %%
from __future__ import annotations

from typing import cast

from traffic.core import Flight, Traffic
from traffic.data import airports, opensky

# %%

t: Traffic = opensky.history(  # type: ignore
    "2021-10-07 12:00",
    "2021-10-07 15:00",
    bounds=(0, 47.5, 5, 50),
)
ground_vehicles = ["392afb", "392b5b", "392adb", "392b1b"]
t = t.query("icao24 not in @ground_vehicles")  # type: ignore
t

# %%


def trim_after_landing(f: Flight) -> None | Flight:
    g = f.aligned_on_ils("LFPG").max("start")
    if g is None:
        return None
    return f.before(g.stop, strict=False)


def d_max(f: Flight) -> bool:
    return f.distance_max >= 59  # type: ignore


lfpg = (
    cast(Traffic, t.query('callsign not in ["FJAVN", "FGLVK"]'))
    .has("aligned_on_LFPG")
    .pipe(trim_after_landing)
    .distance(airports["LFPG"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="aligned_lfpg.pkl",
    )
)
lfpo = (
    t.has("aligned_on_LFPO")
    .distance(airports["LFPO"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="aligned_lfpo.pkl",
    )
)
lfpb = (
    t.has("aligned_on_LFPB")
    .distance(airports["LFPB"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="aligned_lfpb.pkl",
    )
)
lfob = (
    t.has("aligned_on_LFOB")
    .distance(airports["LFOB"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="aligned_lfob.pkl",
    )
)
lfpv = (
    t.has("aligned_on_LFPV")
    .distance(airports["LFPV"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="aligned_lfpv.pkl",
    )
)
landing = lfpg + lfpo + lfpb + lfob + lfpv


# %%


def clip(f: Flight) -> None | Flight:
    g = f.aligned_on_runway("LFPG").max("duration")
    if g is None:
        return f
    return f.after(g.stop)


lfpg = (
    cast(Traffic, t.query("icao24 not in ['4cc0df', '3c4aa2']"))
    .takeoff_from("LFPG")
    .distance(airports["LFPG"])
    .query("distance < 60")
    .filter_if(d_max)
    .pipe(clip)
    .eval(
        desc="",
        max_workers=4,
        cache_file="takeoff_lfpg.pkl",
    )
)
lfpo = (
    t.takeoff_from("LFPO")
    .distance(airports["LFPO"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="takeoff_lfpo.pkl",
    )
)
lfpb = (
    t.takeoff_from("LFPB")
    .distance(airports["LFPB"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="takeoff_lfpb.pkl",
    )
)
lfob = (
    t.takeoff_from("LFOB")
    .distance(airports["LFOB"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="takeoff_lfob.pkl",
    )
)
lfpv = (
    t.takeoff_from("LFPV")
    .distance(airports["LFPV"])
    .query("distance < 60")
    .filter_if(d_max)
    .eval(
        desc="",
        max_workers=4,
        cache_file="takeoff_lfpv.pkl",
    )
)
takeoff = lfpg + lfpo + lfpb + lfob + lfpv
# %%

# remove too many points for QQE940
dataset = (
    (
        landing
        + takeoff.query('callsign != "QQE940"')
        + takeoff["QQE940"].query("groundspeed.notnull()")
        + t["FHMAC"]
    )
    .drop(
        columns=[
            "alert",
            "geoaltitude",
            "hour",
            "last_position",
            "spi",
            "distance",
        ]
    )
    .reset_index(drop=True)
    .drop_duplicates()
)

dataset.to_json("quickstart_raw.json.gz", orient="records")
