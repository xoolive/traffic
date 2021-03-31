# %%

import sys
from pathlib import Path
from typing import List

import altair as alt
import pandas as pd

from traffic.data import airports

alt.renderers.set_embed_options(actions=False)

# %%

dataset_path = sys.argv[1]
dates = [
    "20200101",
    "20200201",
    "20200301",
    "20200401",
    "20200501",
    "20200601",
    "20200701",
    "20200801",
    "20200901",
    "20201001",
    "20201101",
    "20201201",
]
flightlist = pd.concat(
    pd.read_csv(
        # only take the latest version of the file
        max(Path(dataset_path).glob(f"flightlist_{date}_*.csv.gz")),
        parse_dates=["firstseen", "lastseen", "day"],
    )
    for date in dates
)

# %%

airlines_subset = [
    # European airlines
    ["AFR", "KLM", "BAW", "DLH", "AZA", "IBE", "SWR"],
    # American airlines
    ["AAL", "ACA", "DAL", "UAL", "LAN"],
    # Asian airlines
    ["AIC", "JAL", "CPA", "QFA", "SIA", "KAL", "ANA", "UAE", "ANZ"],
    # Few low-cost airlines
    ["EZY", "VLG", "TRA", "RYR", "ROU", "AXM", "WZZ", "APJ", "JST"],
    # Cargo
    ["FDX", "UPS", "GTI", "CLX", "GEC"],
]

data = pd.concat(
    (
        flightlist.query(f'callsign.str.startswith("{airline}")')
        .groupby("day")
        .agg(dict(callsign="count"))
        .rename(columns=dict(callsign=airline))
        for airline in sum(airlines_subset, [])
    ),
    axis=1,
).fillna(0)

source = (
    data.reset_index()
    .melt("day", var_name="airline", value_name="count")
    .groupby("airline", as_index=False)
    .apply(
        lambda x: x.assign(
            max_=lambda df: df["count"][:30].mean(),
            rate=lambda df: df["count"] / df.max_,
        )
    )[["day", "airline", "count", "rate"]]
)


def airline_chart(
    source: alt.Chart, subset: List[str], name: str, loess=True
) -> alt.Chart:

    chart = source.transform_filter(
        alt.FieldOneOfPredicate(field="airline", oneOf=subset)
    )

    highlight = alt.selection(
        type="single", nearest=True, on="mouseover", fields=["airline"]
    )

    points = (
        chart.mark_point()
        .encode(
            x="day",
            y=alt.Y("rate", title="# of flights (normalized)"),
            color=alt.Color("airline", legend=alt.Legend(title=name)),
            tooltip=["day", "airline", "count"],
            opacity=alt.value(0.3),
        )
        .add_selection(highlight)
    )

    lines = chart.mark_line().encode(
        x="day",
        y="rate",
        color="airline",
        size=alt.condition(~highlight, alt.value(1), alt.value(3)),
    )
    if loess:
        lines = lines.transform_loess(
            "day", "rate", groupby=["airline"], bandwidth=0.2
        )

    return lines + points


chart = alt.Chart(source)
result = alt.vconcat(
    *[
        airline_chart(chart, airline_, name, "FDX" not in airline_).properties(
            width=550, height=150
        )
        for name, airline_ in zip(
            [
                "European airlines",
                "American airlines",
                "Asian airlines",
                "Low-cost airlines",
                "Cargo airlines",
            ],
            airlines_subset,
        )
    ]
).resolve_scale(color="independent")

result.save("covid19_airlines.json", indent=2)

# %%

airports_subset = [
    # Europe
    ["LFPG", "EGLL", "EHAM", "EDDF", "LEMD", "LIRF", "LSZH", "UUEE"],
    # Eastern Asia
    ["VHHH", "RJBB", "RJTT", "RKSI", "RCTP", "RPLL"],
    # Asia/Pacific
    ["YSSY", "YMML", "OMDB", "VABB", "VIDP", "WSSS"],
    # Americas
    ["CYYZ", "KSFO", "KLAX", "KATL", "KJFK", "SBGR"],
]


data = pd.concat(
    (
        flightlist.query(f'origin == "{airport}"')
        .groupby("day")
        .agg(dict(callsign="count"))
        .rename(columns=dict(callsign=airport))
        for airport in sum(airports_subset, [])
    ),
    axis=1,
).fillna(0)

source = (
    data.reset_index()
    .melt("day", var_name="airport", value_name="count")
    .merge(
        airports.data[["icao", "municipality"]],
        left_on="airport",
        right_on="icao",
        how="left",
    )[["day", "airport", "count", "municipality"]]
    .rename(columns=dict(municipality="city"))
)


def airport_chart(source: alt.Chart, subset: List[str], name: str) -> alt.Chart:

    chart = source.transform_filter(
        alt.FieldOneOfPredicate(field="airport", oneOf=subset)
    )

    highlight = alt.selection(
        type="single", nearest=True, on="mouseover", fields=["airport"]
    )

    points = (
        chart.mark_point()
        .encode(
            x="day",
            y=alt.Y("count", title="# of departing flights"),
            color=alt.Color("airport", legend=alt.Legend(title=name)),
            tooltip=["day", "airport", "city", "count"],
            opacity=alt.value(0.3),
        )
        .add_selection(highlight)
    )

    lines = (
        chart.mark_line()
        .encode(
            x="day",
            y="count",
            color="airport",
            size=alt.condition(~highlight, alt.value(1), alt.value(3)),
        )
        .transform_loess("day", "count", groupby=["airport"], bandwidth=0.2)
    )

    return lines + points


chart = alt.Chart(source)
result = alt.vconcat(
    *[
        airport_chart(chart, airport_, name).properties(width=550, height=150)
        for name, airport_ in zip(
            [
                "European airports",
                "East-Asian airports",
                "Asian/Australian airports",
                "American airports",
            ],
            airports_subset,
        )
    ]
).resolve_scale(color="independent")

result.save("covid19_airports.json")


# %%
