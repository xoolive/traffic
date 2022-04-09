How to access airspace information?
===================================

Each Air Control Center (ACC), in charge of providing air traffic control
services to controlled flights within its airspace, is subdivided into
elementary sectors that are used or combined to build control sectors operated
by a pair of air traffic controllers.

Airspace sectorisation consists in partitioning the overall ACC airspace into a
given number of these control sectors. In most centres, the set of control
sectors deployed, i.e. the sector configuration, varies throughout the day.
Basically, sectors are split when controllers’ workload increases, and merged
when it decreases.

An `Airspace <../traffic.core.airspace.html>`__ is represented as a set of
polygons of geographical coordinates, extruded to a lower and upper altitudes. 
Airspaces can be of different types, corresponding to different kinds of
activities allowed (civil, military, glider, etc.) or services provided.

- | a **Flight Information Region (FIR)** is the largest regular division of
    airspace in use in the world today, roughly corresponding to a country,
    although some FIRs encompass several countries, and larger countries are
    subdivided into a number of regional FIRs.
  | Read more about `FIRs of the world
    <https://observablehq.com/@xoolive/flight-information-regions>`_
- | a **Terminal Maneuvering Area (TMA)** is usually controlled by the airport it
    is attached to: this airspace is usually reserved for aircraft landing at or
    taking off from that airport;
- | an **Air Route Traffic Control Center (ARTCC)** or an **Air Control Center
    (ACC)** represents the area that is controlled by people located in the same
    center;
- | an **elementary sector** is usually the smallest subdivision of airspace that can
    be controlled by a pair of air traffic controllers.

FIRs from countries in the EUROCONTROL area
-------------------------------------------

Airspaces are usually handled as a wrapper around `GeoDataFrame
<https://geopandas.readthedocs.io/>`_, i.e.  Pandas data frames with a
particular geometry feature.

.. jupyter-execute::

   from traffic.data import eurofirs

   eurofirs.head()

As FIRs usually consist of a single polygon with an upper and a lower limit (in
Flight Level, i.e. FL 300 corresponds to 30,000 ft), their representation is
simple in a Jupyter environment.

.. jupyter-execute::

   eurofirs["ENOB"]

Leaflet representations are also easily accessible:

.. jupyter-execute::

   eurofirs["EKDK"].map_leaflet()

The following snippet plots the whole dataset:

.. jupyter-execute::

   import matplotlib.pyplot as plt

   from cartes.crs import LambertConformal
   from cartes.utils.features import countries, lakes, ocean

   fig, ax = plt.subplots(
       figsize=(15, 10),
       subplot_kw=dict(projection=LambertConformal(10, 45)),
   )

   ax.set_extent((-20, 45, 25, 75))

   ax.add_feature(countries(scale="50m"))
   ax.add_feature(lakes(scale="50m"))
   ax.add_feature(ocean(scale="50m"))

   ax.spines['geo'].set_visible(False)

   for fir in eurofirs:
       fir.plot(ax, edgecolor="#3a3aaa")
       if fir.designator not in ["ENOB", "LPPO", "GCCC"] and fir.area > 1e11:
           fir.annotate(
               ax,
               s=fir.designator,
               ha="center",
               color="#3a3aaa",
               fontname="Ubuntu",
               fontsize=13,
           )

FIRs and ARTCC from the FAA
----------------------------

The Federal Aviation Administration (FAA) publishes some data about their
airspace in open data. The data is automatically downloaded the first time
you try to access it.

Find more about this service `here <https://adds-faa.opendata.arcgis.com/>`_.
On the following map, Air Route Traffic Control Centers (ARTCC) are displayed
together with neighbouring FIRs.

.. jupyter-execute::

   from traffic.data.faa import airspace_boundary


.. jupyter-execute::

   import matplotlib.pyplot as plt

   from cartes.crs import AzimuthalEquidistant
   from cartes.utils.features import countries

   fig, ax = plt.subplots(
       figsize=(10, 10),
       subplot_kw=dict(projection=AzimuthalEquidistant(central_longitude=-100)),
   )

   ax.add_feature(countries(scale="50m"))
   ax.set_extent((-130, -65, 15, 60))
   ax.spines['geo'].set_visible(False)


   for airspace in airspace_boundary.query(
       'type == "FIR" and designator.str[0] in ["C", "M", "T"]'
   ):
       airspace.plot(ax, edgecolor="#f58518", lw=3, alpha=0.5)

   for airspace in airspace_boundary.query(
       'type == "ARTCC" and designator != "ZAN"'  # Anchorage
   ).at_level(100):
       airspace.plot(ax, edgecolor="#4c78a8", lw=2)
       airspace.annotate(
           ax,
           s=airspace.designator,
           color="#4c78a8",
           ha="center",
           fontname="Ubuntu",
           fontsize=14,
       )



Airspace data from EUROCONTROL
------------------------------

.. warning::

   Access conditions and configuration for these sources of data is detailed
   `here <eurocontrol.html>`_.

.. tip::

   According to the source of data you may have access to, different parsers are
   implemented but they expose the same API.  Data is usually not exactly
   consistent (coordinates, types, etc.) but you should still be able to safely
   replace ``nm_airspaces`` with ``aixm_airspaces`` in the following examples.

.. jupyter-execute::

   from traffic.data import nm_airspaces

Get a single airspace
~~~~~~~~~~~~~~~~~~~~~

In these files, airspaces being a composition of elementary airspaces are
widespread. Their union is computed and yields a list of polygons associated
with minimum and maximum flight levels.

.. jupyter-execute::

   nm_airspaces["EDYYUTAX"]

The Leaflet view and the Matplotlib view, flatten the polygon prior to displaying it:

.. jupyter-execute::

   nm_airspaces["EDYYUTAX"].map_leaflet()

Get many airspaces
~~~~~~~~~~~~~~~~~~

Airspaces are stored as a GeoDataFrame, so all pandas operators may be applied
to get a subset of them, for example based on their types or designator.

All available airspace types can be accessed here:

.. jupyter-execute::

   nm_airspaces.data["type"].unique()

In the following examples, we get all FIR spaces in the LF domain (France). You
may notice that a single designator may be represented by several entries in the
GeoDataFrame, but the representation of each shape you get through iteration,
indexation or ``__geo_interface__`` attribute  (used in Altair) is properly
computed.

.. jupyter-execute::

   france_fir = nm_airspaces.query('type == "FIR" and designator.str.startswith("LF")')
   france_fir.head(10)

.. jupyter-execute::

   import altair as alt

   from cartes.atlas import europe

   france = alt.Chart(europe.topo_feature).transform_filter(
       "datum.properties.geounit == 'France'"
   )

   base = (
       alt.Chart(france_fir)
       .mark_geoshape(stroke="white")
       # In this dataset, UIR and FIR are both tagged as FIR
       # => we reconstruct the type and designator
       .transform_calculate(suffix="slice(datum.designator, -3)")
       .transform_calculate(designator="slice(datum.designator, 0, -3)")
       .properties(width=300)
   )

   chart = (
       alt.concat(
           alt.layer(
               base.encode(alt.Color("designator:N", title="FIR"))
               .transform_filter("datum.suffix == 'FIR'"),
               france.mark_geoshape(stroke="#ffffff", strokeWidth=3, fill="#ffffff00"),
               france.mark_geoshape(stroke="#79706e", strokeWidth=1, fill="#ffffff00"),
               base.mark_text(fontSize=14, font="Lato")
               .encode(
                   alt.Latitude("latitude:Q"),
                   alt.Longitude("longitude:Q"),
                   alt.Text("designator:N"),
               )
               .transform_filter("datum.suffix == 'FIR'"),
           ),
           alt.layer(
               base.encode(alt.Color("designator:N"))
               .transform_filter("datum.suffix == 'UIR'"),
               france.mark_geoshape(stroke="#ffffff", strokeWidth=3, fill="#ffffff00"),
               france.mark_geoshape(stroke="#79706e", strokeWidth=1, fill="#ffffff00"),
           )
       )
       .configure_legend(orient="bottom", labelFontSize=12, titleFontSize=12)
       .configure_view(stroke=None)
   )

   chart

.. warning::

    Note that the FIR and UIR boundaries do not presume anything about how air
    traffic control is organised in that area. See below a map of the areas
    controlled by different en-route air traffic control centres in France.

.. jupyter-execute::


    centres = ["LFBBBDX", "LFRRBREST", "LFEERMS", "LFFFPARIS", "LFMMRAW", "LFMMRAE"]
    subset = (
        nm_airspaces.query(f"designator in {centres}")
        .replace("LFMMRAW", "Aix-en-Provence (Sud-Est)")
        .replace("LFMMRAE", "Aix-en-Provence (Sud-Est)")
        .replace("LFBBBDX", "Bordeaux (Sud-Ouest)")
        .replace("LFEERMS", "Reims (Est)")
        .replace("LFFFPARIS", "Athis-Mons (Nord)")
        .replace("LFRRBREST", "Brest (Ouest)")
    )


    chart = (
        alt.hconcat(
            alt.layer(
                alt.Chart(subset.at_level(220))
                .mark_geoshape(stroke="white")
                .encode(
                    alt.Tooltip(["designator:N"]),
                    alt.Color("designator:N", title="CRNA"),
                ),
                france.mark_geoshape(stroke="#ffffff", strokeWidth=3, fill="#ffffff00"),
                france.mark_geoshape(stroke="#79706e", strokeWidth=1, fill="#ffffff00"),
            ).properties(title="FL 220", width=300),
            alt.layer(
                alt.Chart(subset.at_level(320))
                .mark_geoshape(stroke="white")
                .encode(
                    alt.Tooltip(["designator:N"]),
                    alt.Color("designator:N", title="CRNA"),
                ),
                france.mark_geoshape(stroke="#ffffff", strokeWidth=3, fill="#ffffff00"),
                france.mark_geoshape(stroke="#79706e", strokeWidth=1, fill="#ffffff00"),
            ).properties(title="FL320", width=300),
        )
        .configure_legend(orient="bottom", labelFontSize=12, titleFontSize=12)
        .configure_view(stroke=None)
        .configure_title(fontSize=16, font="Lato", anchor="start")
    )

    chart

Iterate through many airspaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following trick lets you return the largest CS sector with a designator
starting with LFBB (Bordeaux FIR/ACC).

.. jupyter-execute::

   # Find the largest CS in Bordeaux ACC
   from operator import attrgetter

   max(
       nm_airspaces.query('type == "CS" and designator.str.startswith("LFBB")'),
       key=attrgetter("area"),  # equivalent to `lambda x: x.area`
   )

Free Route Areas (FRA) from EUROCONTROL
---------------------------------------

The Free Route information consists of regular airspace information:


.. jupyter-execute::

   from traffic.data import nm_freeroute
   nm_freeroute

.. jupyter-execute::

   nm_freeroute["BOREALIS"].map_leaflet(zoom=3)

However, in addition to these airspace, there is a database of points attached to a Free Route Area:

- ``I`` refer to **I** ntermediate points in the middle of an airspace;
- ``E`` and ``X`` mark points of **E** ntry and e **X** it;
- ``A`` and ``D`` refer to **A** rrival and **D** eparture and are attached to an airport.

.. jupyter-execute::

   nm_freeroute.points.query('FRA == "BOREALIS"')
