# %%
import altair as alt
import pandas as pd
import requests

c = requests.get("https://opensky-network.org/api/stats/facts?extended=true")
data = c.json()

# %%
df = pd.DataFrame.from_records(
    data["Message Counts"], columns=["timestamp", "daily", "cumulative"]
)
df = df.assign(timestamp=lambda df: pd.to_datetime(df.timestamp, unit="ms"))

# %%
chart = alt.Chart(df).encode(
    x=alt.X("timestamp", title=None),
    y=alt.Y(
        "daily", axis=alt.Axis(format="~s", title="Number of daily messages"),
    ),
)

annotation = (
    alt.Chart(
        pd.DataFrame.from_records(
            [
                {
                    "timestamp": pd.Timestamp("2020-03-25"),
                    "daily": 5.5e9,
                    "text": "COVID-19",
                },
                {
                    "timestamp": pd.Timestamp("2019-02-15"),
                    "daily": 20e9,
                    "text": "Stopped anonymous feeding",
                },
            ]
        )
    )
    .encode(x="timestamp", y="daily", text="text")
    .mark_text(align="right", baseline="middle", fontSize=14, fontWeight="bold")
)

view = (
    (
        chart.mark_line(opacity=0.0,)
        + annotation
        + chart.transform_loess("timestamp", "daily", bandwidth=0.02).mark_line(
            color="#d35400"
        )
    )
    .configure(font="Ubuntu",)
    .properties(
        width=500,
        height=300,
        title="Number of daily messages received by the OpenSky network",
    )
    .configure_title(fontSize=15, font="Ubuntu", anchor="start", color="gray")
    .configure_axis(
        labelFontSize=14,
        titleFontSize=14,
        labelFont="Ubuntu",
        titleFont="Ubuntu",
    )
)

view

# %%
