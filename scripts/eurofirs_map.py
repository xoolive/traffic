import matplotlib.pyplot as plt
from cartopy.crs import TransverseMercator

from traffic.data import eurofirs
from traffic.drawing import countries, lakes, ocean, rivers

fig, ax = plt.subplots(
    figsize=(15, 10), subplot_kw=dict(projection=TransverseMercator(10, 45))
)

ax.set_extent((-20, 45, 30, 70))

ax.add_feature(countries(scale="50m"))
ax.add_feature(rivers(scale="50m"))
ax.add_feature(lakes(scale="50m"))
ax.add_feature(ocean(scale="50m"))

for key, fir in eurofirs.items():
    fir.plot(ax, edgecolor="#3a3aaa", lw=2, alpha=0.5)
    if key not in ["ENOB", "LPPO", "GCCC"]:
        fir.annotate(
            ax,
            s=key,
            ha="center",
            color="#3a3aaa",
            fontname="Ubuntu",
            fontsize=13,
        )

fig.savefig("eurofirs.png", bbox_inches="tight", pad_inches=0, dpi=200)
