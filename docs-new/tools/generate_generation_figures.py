"""Generate the static figures used by the trajectory generation guide.

Run from the repository root with:

    uv run --no-sync --with scikit-learn \
        docs-new/tools/generate_generation_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from cartes.crs import EuroPP
from cartopy.crs import PlateCarree
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from traffic.algorithms.generation import Generation
from traffic.data import airports
from traffic.data.datasets import landing_zurich_2019

OUTPUT_DIR = Path(__file__).parents[1] / "docs" / "assets" / "images"
PROJECTION = EuroPP()


def elapsed_seconds(frame: pd.DataFrame) -> pd.Series:
    return (frame.timestamp - frame.timestamp.min()).dt.total_seconds()


def prepare_traffic():
    traffic = (
        landing_zurich_2019.query(
            "runway == '14' and initial_flow == '162-216'"
        )
        .assign_id()
        .unwrap()
        .resample(100)
        .eval()
    )[:1000]
    return (
        traffic.compute_xy(projection=PROJECTION)
        .iterate_lazy()
        .assign(timedelta=elapsed_seconds)
        .eval()
    )


def plot_traffic(ax, traffic, *, color: str, alpha: float) -> None:
    for _, frame in traffic.data.groupby("flight_id", sort=False):
        ax.plot(
            frame.x / 1000,
            frame.y / 1000,
            color=color,
            alpha=alpha,
            linewidth=0.7,
        )


def style_map(ax, title: str) -> None:
    ax.set_title(title, loc="left")
    ax.set_xlabel("projected x (km)")
    ax.set_ylabel("projected y (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)


def position_generation_figure(traffic) -> None:
    generator = Generation(
        generation=GaussianMixture(
            n_components=2,
            covariance_type="diag",
            random_state=42,
        ),
        features=["x", "y", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
    ).fit(traffic)
    generated = generator.sample(150, projection=PROJECTION)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    plot_traffic(axes[0], traffic, color="#607d8b", alpha=0.035)
    plot_traffic(axes[1], generated, color="#1976d2", alpha=0.12)
    style_map(axes[0], "Training trajectories")
    style_map(axes[1], "150 generated trajectories")
    fig.savefig(
        OUTPUT_DIR / "trajectory-generation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def reconstruction_direction_figure(traffic) -> None:
    generator = Generation(
        generation=GaussianMixture(
            n_components=2,
            covariance_type="diag",
            random_state=42,
        ),
        features=["track", "groundspeed", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1)),
    ).fit(traffic)

    # Reuse 40 complete feature profiles from the fitted matrix. This keeps
    # the comparison focused on reconstruction direction rather than on the
    # quality of a particular generative model.
    prepared = generator.prepare_features(traffic)
    indices = (
        pd.Series(range(len(prepared))).sample(40, random_state=42).to_numpy()
    )
    samples = prepared[indices]
    if generator.scaler is not None:
        samples = generator.scaler.inverse_transform(samples)

    selected_flights = [traffic[int(index)] for index in indices]
    initial_points = pd.DataFrame(
        [
            flight.data.sort_values("timestamp").iloc[0]
            for flight in selected_flights
        ]
    )
    final_points = pd.DataFrame(
        [
            flight.data.sort_values("timestamp").iloc[-1]
            for flight in selected_flights
        ]
    )
    approach_reference = {
        "latitude": float(initial_points.latitude.median()),
        "longitude": float(initial_points.longitude.median()),
    }
    runway_reference = {
        "latitude": float(final_points.latitude.median()),
        "longitude": float(final_points.longitude.median()),
    }

    reconstructed_forward = generator.build_traffic(
        samples.copy(), coordinates=approach_reference, forward=True
    )
    reconstructed_backward = generator.build_traffic(
        samples.copy(), coordinates=runway_reference, forward=False
    )

    forward_xy = reconstructed_forward.compute_xy(projection=PROJECTION)
    backward_xy = reconstructed_backward.compute_xy(projection=PROJECTION)
    forward_groups = forward_xy.data.sort_values("timestamp").groupby(
        "flight_id", sort=False
    )
    backward_groups = backward_xy.data.sort_values("timestamp").groupby(
        "flight_id", sort=False
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    overview_panels = [
        (
            forward_xy,
            forward_groups.tail(1),
            forward_groups.head(1).iloc[0],
            "forward=True: reference is the start",
            "unconstrained end",
        ),
        (
            backward_xy,
            backward_groups.head(1),
            backward_groups.tail(1).iloc[0],
            "forward=False: reference is the end",
            "unconstrained start",
        ),
    ]
    for ax, (reconstructed, free_points, reference, title, free_label) in zip(
        axes, overview_panels
    ):
        plot_traffic(ax, reconstructed, color="#1976d2", alpha=0.22)
        ax.scatter(
            free_points.x / 1000,
            free_points.y / 1000,
            s=10,
            color="#d32f2f",
            alpha=0.65,
            label=free_label,
            zorder=3,
        )
        ax.scatter(
            [reference.x / 1000],
            [reference.y / 1000],
            marker="*",
            s=160,
            color="#111111",
            label="fixed reference",
            zorder=4,
        )
        style_map(ax, title)
        ax.legend(frameon=False, loc="best")

    fig.savefig(
        OUTPUT_DIR / "trajectory-reconstruction-direction-overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    airport = airports.source("ourairports")["LSZH"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": PROJECTION},
    )
    panels = [
        (
            reconstructed_forward,
            "forward=True: final positions are free",
        ),
        (
            reconstructed_backward,
            "forward=False: final position is fixed",
        ),
    ]
    for ax, (reconstructed, title) in zip(axes, panels):
        airport.plot(
            ax,
            footprint=False,
            runways={"color": "#111111", "linewidth": 2, "zorder": 4},
            labels=False,
        )
        for _, frame in reconstructed.data.groupby("flight_id", sort=False):
            ax.plot(
                frame.longitude,
                frame.latitude,
                transform=PlateCarree(),
                color="#1976d2",
                alpha=0.25,
                linewidth=0.8,
                zorder=2,
            )
        ax.set_extent(airport, buffer=0.04)
        ax.set_title(title, loc="left")
        ax.spines["geo"].set_visible(False)

    forward_ends = forward_groups.tail(1)
    axes[0].scatter(
        forward_ends.longitude,
        forward_ends.latitude,
        transform=PlateCarree(),
        s=14,
        color="#d32f2f",
        alpha=0.75,
        label="unconstrained final position",
        zorder=6,
    )
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].scatter(
        [runway_reference["longitude"]],
        [runway_reference["latitude"]],
        transform=PlateCarree(),
        marker="*",
        s=180,
        color="#d32f2f",
        label="fixed RWY 14 reference",
        zorder=6,
    )
    axes[1].legend(frameon=False, loc="lower left")

    fig.savefig(
        OUTPUT_DIR / "trajectory-reconstruction-direction.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(
        "Approach reference: "
        f"latitude={approach_reference['latitude']:.5f}, "
        f"longitude={approach_reference['longitude']:.5f}"
    )
    print(
        "RWY 14 reference: "
        f"latitude={runway_reference['latitude']:.5f}, "
        f"longitude={runway_reference['longitude']:.5f}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traffic = prepare_traffic()
    position_generation_figure(traffic)
    reconstruction_direction_figure(traffic)


if __name__ == "__main__":
    main()
