from itertools import count

import matplotlib.pyplot as plt
from cartes.crs import Mercator  # type: ignore
from cartes.osm import Overpass


def main() -> None:

    icaos = [
        "EGLL",  # Heathrow Airport
        "LFPG",  # Charles de Gaulle Airport
        "EHAM",  # Amsterdam Airport Schiphol
        "EDDF",  # Frankfurt am Main Airport
        "LEMD",  # Adolfo Suárez Madrid–Barajas Airport
        "LEBL",  # Josep Tarradellas Barcelona–El Prat Airport
        "LTFM",  # Istanbul Airport
        "UUEE",  # Sheremetyevo International Airport
        "EDDM",  # Munich Airport
        "EGKK",  # Gatwick Airport
        "LIRF",  # Leonardo da Vinci–Fiumicino Airport
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
        nrows=4,
        ncols=5,
        figsize=(118.9 * cm, 84.1 * cm),  # A0
    )

    for i, ax, ap in zip(count(), axs.flat, icaos):
        airport = Overpass.request(area=dict(icao=ap), aeroway=True)
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
        # ax.set_title(ap)
        ax.text(
            (i % 5 + 0.9) / 6,
            4.5 / 5 - (i // 5) / 5,
            ap,
            transform=fig.transFigure,
            fontSize=40,
            fontFamily="B612",
            color="0.1",
        )

    plt.savefig("europe_airports.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
