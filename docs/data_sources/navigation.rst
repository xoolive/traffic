How to access beacons and airways information?
==============================================

Air navigation and communication with air traffic control (ATC) is mostly based
on the use of navigational beacons and airways. In recent years, airways tend to
be replaced by Free Route Areas (FRA), but airways are still bound to exist in
some parts of the world, Oceanic areas, mountainous areas, and free route
navigation remains based on navigational beacons.

Navigational beacons (called ``navaids`` in the library) are of different types:
historically the first ones were attached to VOR equipment, but we can mostly
consider they are a name attached to geographic coordinates.

Basic data of questionable accuracy is provided with the library:

.. code:: python

    from traffic.data import airways, navaids

However, if other sources of data are configured (e.g. `EUROCONTROL data files
<eurocontrol.html>`_), the library will look into all sources (with the basic
data source set as lowest priority)


.. autoclass:: traffic.data.basic.navaid.Navaids
    :members:
    :no-inherited-members:
    :no-undoc-members:

.. autoclass:: traffic.data.basic.airways.Airways
    :members:
    :no-inherited-members:
    :no-undoc-members:

.. tip::

    The same L888 route can also be plotted inside a Leaflet widget.

.. jupyter-execute::

    from ipyleaflet import Map, basemaps

    m = Map(center=(32.3, 99), zoom=4, basemap=basemaps.Stadia.StamenTerrain)
    m.add(airways["L888"])

    m
