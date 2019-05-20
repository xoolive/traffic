FIRs in Europe
--------------

FIR stands for Flight Information Region.

FIR are the largest regular division of airspace in use in the world today.
Every portion of the atmosphere belongs to a specific FIR. Smaller countries'
airspace is encompassed by a single FIR; larger countries' airspace is
subdIvided into a number of regional FIRs.  Some FIRs encompass the territorial
airspace of several countries.  Oceanic airspace is divided into Oceanic
Information Regions and delegated to a controlling authority bordering that
region.

The division among authorities is done by international agreement
through the International Civil Aviation Organization (ICAO). (`source Wikipedia
<https://en.wikipedia.org/wiki/Flight_information_region>`_)


FIRs of countries in the Eurocontrol area are available in the library.

.. code:: python

    from traffic.data import eurofirs
    from traffic.drawing import TransverseMercator, countries, lakes, ocean, rivers

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        1, figsize=(15, 10),
         subplot_kw=dict(projection=TransverseMercator(10, 45))
    )

    ax.add_feature(countries(scale="50m"))
    ax.add_feature(rivers(scale="50m"))
    ax.add_feature(lakes(scale="50m"))
    ax.add_feature(ocean(scale="50m"))

    for name, fir in eurofirs.items():
        fir.plot(ax, edgecolor="#3a3aaa", lw=2, alpha=0.5)

        if name not in ["ENOB", "LPPO", "GCCC"]:
            fir.annotate(
                ax, s=name, ha="center", color="#3a3aaa"
            )

    ax.set_extent((-20, 45, 30, 70))

.. image:: _static/eurofirs.png
   :scale: 25%
   :alt: FIRs in the Eurocontrol area
   :align: center
