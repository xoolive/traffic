import matplotlib.pyplot as plt

from traffic.drawing import countries, rivers, lakes, ocean, TransverseMercator
from traffic.data import eurofirs

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection=TransverseMercator(10, 45))

ax.set_extent((-20, 45, 30, 70))

ax.add_feature(countries(scale="50m"))
ax.add_feature(rivers(scale="50m"))
ax.add_feature(lakes(scale="50m"))
ax.add_feature(ocean(scale="50m"))

for key, fir in eurofirs.items():
    fir.plot(ax, edgecolor="#3a3aaa", lw=2, alpha=.5)
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
