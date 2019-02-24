Graphical user interface
========================

The traffic library comes with a GTK Graphical user interface (GUI) designed
to decode and explore historical and live data. The GUI is accessible through
the following command (clickable application icons are doable and will probably
be automatically generated in future versions)


.. code::

    traffic gui


Data exploration
----------------

The GUI consists of two panes:

- the **display** pane on the left-hand side, with a *map* and a *plots* tab;
- the **command** pane on the right-hand side, with selection and filtering
  buttons.

.. image:: _static/gui_start.png
   :scale: 25 %
   :alt: Startup screen for the GUI
   :align: center

By default, the tool opens with a EuroPP projection. Other projections like
Mercator are also available by default: you can customize more projections in
the configuration file, in the ``[projections]`` section:

.. code::

    [projections]
    default = EuroPP
    extra = Lambert93; Amersfoort; GaussKruger

| Available projections are default cartopy projections, completed by additional common European projections in the cartotools `dependency module <https://github.com/xoolive/cartotools/tree/master/cartotools/crs>`_ (here `Lambert 93 <https://fr.wikipedia.org/wiki/Projection_conique_conforme_de_Lambert#Lambert_93>`_ is the official projection in France, `Amersfoort <https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten>`_ in the Netherlands and `Gauss-Krüger <https://de.wikipedia.org/wiki/Gau%C3%9F-Kr%C3%BCger-Koordinatensystem>`_ in Germany)
| You can implement more projections as plugins or file a `PR in cartotools <https://github.com/xoolive/cartotools/>`_.

You can either pan and zoom the map. Zoom is operated by the mouse or trackpad scrool. Note that on MacOS, the trackpad scroll requires clicking.

In order to explore data, click on *Open file* and select a .pkl file (like ``sample_opensky.pkl`` in the ``data/`` directory) By default, a scatter of all last recorded points is displayed.

.. image:: _static/gui_map.png
   :scale: 25 %
   :alt: Scatter plot
   :align: center

- You may select callsigns in order to plot trajectories.
- Date and altitude sliders operate filters on the full pandas DataFrame.

.. image:: _static/gui_trajectory.png
   :scale: 25 %
   :alt: Trajectories
   :align: center

In the *Plots* tab, you may select one callsign with different signals (e.g.
*altitude* on the left-hand side and *ground speed* on the righ-hand side)
**or (exclusive)** several callsigns with one signal (e.g. *altitude*).

.. image:: _static/gui_plots.png
   :scale: 25 %
   :alt: Trajectories
   :align: center

Data recording
--------------

