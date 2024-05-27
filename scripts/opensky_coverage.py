# %%

from traffic.data import opensky

cov_2023 = opensky.api_global_coverage("2024-05-01")
cov_2018 = opensky.api_global_coverage("2018-01-01")

# %%
import httpx

c = httpx.get("https://opensky-network.org/api/sensor/list")
c.raise_for_status()

import pandas as pd

sensors = pd.DataFrame.from_records(c.json()).assign(
    added=lambda df: pd.to_datetime(df.added, unit="s"),
    lastConnectionEvent=lambda df: pd.to_datetime(
        df.lastConnectionEvent, unit="s"
    ),
    longitude=lambda df: df.position.str["longitude"],
    latitude=lambda df: df.position.str["latitude"],
)

# %%

# %%
import matplotlib.pyplot as plt
from cartes.crs import PlateCarree, Robinson  # type: ignore
from cartes.utils.features import countries

with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        2, 1, figsize=(15, 12), subplot_kw=dict(projection=Robinson())
    )
    for ax_ in ax:
        ax_.add_feature(countries(scale="50m"))
        ax_.set_global()

    cov_2018.plot(ax[0], cmap="Reds", vmin=-18000, vmax=500)
    sensors.query(
        "lastConnectionEvent >= '2018-01-01' and added <= '2018-01-01'"
    ).plot.scatter(
        ax=ax[0],
        x="longitude",
        y="latitude",
        transform=PlateCarree(),
        color="k",
        s=4,
    )
    cov_2023.plot(ax[1], cmap="Reds", vmin=-18000, vmax=500)
    sensors.query(
        "lastConnectionEvent >= '2022-01-01' and added <= '2022-01-01'"
    ).plot.scatter(
        ax=ax[1],
        x="longitude",
        y="latitude",
        transform=PlateCarree(),
        color="k",
        s=4,
    )

    ax[0].set_title("2018", x=0.05, y=0, fontsize=18)
    ax[1].set_title("2024", x=0.05, y=0, fontsize=18)

    fig.savefig("opensky_coverage.png")
# %%
