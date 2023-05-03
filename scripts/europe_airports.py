from itertools import count

import matplotlib.pyplot as plt
from cartes.crs import Mercator
from cartes.osm import Overpass


def main() -> None:
    nrows = 5
    ncols = 4

    icaos = [
        "EGLL",  # Heathrow Airport
        "LFPG",  # Charles de Gaulle Airport
        "EHAM",  # Amsterdam Airport Schiphol
        "EDDF",  # Frankfurt am Main Airport
        "LEMD",  # Adolfo Suárez Madrid-Barajas Airport
        "LEBL",  # Josep Tarradellas Barcelona-El Prat Airport
        "LTFM",  # Istanbul Airport
        "UUEE",  # Sheremetyevo International Airport
        "EDDM",  # Munich Airport
        "EGKK",  # Gatwick Airport
        "LIRF",  # Leonardo da Vinci-Fiumicino Airport
        "EIDW",  # Dublin Airport
        "LFPO",  # Orly Airport
        "LOWW",  # Vienna International Airport
        "LSZH",  # Zurich Airport
        "LPPT",  # Lisbon Airport
        "EKCH",  # Copenhagen Airport
        "LEPA",  # Palma de Mallorca Airport
        "EGCC",  # Manchester Airport
        "LIMC",  # Milan Malpensa Airport
    ]

    cm = 1 / 2.54  # centimeters in inches
    fig, axs = plt.subplots(
        subplot_kw=dict(projection=Mercator()),
        nrows=nrows,
        ncols=ncols,
        # figsize=(118.9 * cm, 84.1 * cm),  # A0
        figsize=(59.45 * cm, 84.1 * cm),  # A1
    )

    for i, ax, icao in zip(count(), axs.flat, icaos):
        airport = Overpass.request(area=dict(icao=icao), aeroway=True)
        airport.plot(
            ax,
            by="aeroway",
            aerodrome=dict(alpha=0),  # mask contour
            gate=dict(alpha=0),  # mute
            parking_position=dict(alpha=0),  # mute
            tower=dict(markersize=500),  # reduce
            jet_bridge=dict(color="0.3"),  # change color
            navigationaid=dict(
                papi=dict(alpha=0),
                glidepath=dict(alpha=0),
                approach_light=dict(alpha=0),
            ),  # mute
            windsock=dict(markersize=100),
            helipad=dict(markersize=350),
            runway=dict(color="sienna"),
        )
        ax.spines["geo"].set_visible(False)
        ax.text(
            (i % ncols + 0.9) / (ncols + 1),
            (nrows + 0.4) / (nrows + 1) - (i // ncols) / (nrows + 1),
            icao,
            transform=fig.transFigure,
            fontsize=40,
            font="B612",
            color="0.1",
        )

    plt.savefig("europe_airports.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
