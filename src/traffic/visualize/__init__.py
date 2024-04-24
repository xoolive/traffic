# ruff: noqa: E501

from pathlib import Path
from typing import Any

import matplotlib as mpl

_traffic_style = """
figure.figsize: 10, 7
figure.edgecolor: white
figure.facecolor: white

lines.linewidth: 1.5
lines.markeredgewidth: 0
lines.markersize: 10
lines.dash_capstyle: butt

legend.fancybox: True
font.size: 13

axes.prop_cycle: cycler('color', ['4c78a8', 'f58518', '54a24b', 'b79a20', '439894', 'e45756', 'd67195', 'b279a2', '9e765f', '7970ce'])
axes.linewidth: 0
axes.titlesize: 16
axes.labelsize: 14

xtick.labelsize: 14
ytick.labelsize: 14
xtick.major.size: 0
xtick.minor.size: 0
ytick.major.size: 0
ytick.minor.size: 0

axes.grid: True
grid.alpha: 0.3
grid.linewidth: 0.5
grid.linestyle: -
grid.color: 0e1111

savefig.bbox: tight
savefig.format: png
"""

config_dir = mpl.get_configdir()
mpl_style_location = Path(f"{config_dir}/stylelib/traffic.mplstyle")
if not mpl_style_location.parent.is_dir():  # coverage: ignore
    mpl_style_location.parent.mkdir(parents=True)

if not mpl_style_location.exists():
    # keep the slow import here
    import matplotlib.pyplot as plt

    mpl_style_location.write_text(_traffic_style)
    plt.style.reload_library()


def __getattr__(name: str) -> Any:  # coverage: ignore
    msg = f"module {__name__} has no attribute {name}"

    if name.startswith("_"):
        raise AttributeError(msg)

    import cartes.crs
    import cartes.utils.features

    if name in dir(cartes.crs):
        return getattr(cartes.crs, name)

    if name in dir(cartes.utils.features):
        return getattr(cartes.utils.features, name)

    raise AttributeError(msg)
