# flake8: noqa

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .cartopy import *

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

axes.prop_cycle: cycler('color', ['779ae3', 'e77074', '74c863', 'c2c55e', 'd27bcd', '60ceaf', 'd49146'])
axes.linewidth: 0
axes.titlesize: 22
axes.labelsize: 16

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

savefig.transparent: True
savefig.bbox: tight
savefig.format: png
"""

config_dir = mpl.get_configdir()
mpl_style_location = Path(f"{config_dir}/stylelib/traffic.mplstyle")
if not mpl_style_location.parent.is_dir():
    mpl_style_location.parent.mkdir(parents=True)
mpl_style_location.write_text(_traffic_style)

plt.style.reload_library()
