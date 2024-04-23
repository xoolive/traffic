Visualize trajectories with Leaflet
===================================

Data visualization is integrated with the `ipyleaflet
<https://ipyleaflet.readthedocs.io/>`__ library as long as it is installed as an
optional dependency.

You may install `traffic` with the `leaflet` option, the `full` option, or
install ipyleaflet and ipywidgets manually:

.. code:: bash

    # at install time
    pip install traffic[leaflet]  # or full

    # with poetry
    poetry install traffic -E leaflet  # or -E full

    # or simply manually
    pip install ipyleaflet ipywidgets
    conda install -c conda-forge ipyleaflet ipywidgets


Flights can easily be displayed in a Leaflet widget:

.. jupyter-execute::

    from traffic.data.samples import belevingsvlucht

    belevingsvlucht.map_leaflet()

The option for the airport centres the view on the airport and highlights the runways:

.. jupyter-execute::

    belevingsvlucht.map_leaflet(airport="EHAM", zoom=12)

There is also a highlight parameter to put colours on part of the trajectory.
The dictionary passed in parameter takes colours as keys (HTML values) and
strings or callbacks as parameters.

It is also possible to add routes, points, flight plans etc. with the ``.add`` method:

.. jupyter-execute::

    from traffic.data import airports

    m = belevingsvlucht.map_leaflet(
        zoom=8,
        highlight={
            "red": 'aligned_on_ils("EHAM")',
            "#bd0026": lambda flight: flight.aligned_on_ils("EHLE"),
            "#feb24c": "holding_pattern",
        }
    )

    m.add(airports["EHLE"].point)
    m

It is also possible to call :meth:`~traffic.core.Traffic.map_leaflet()` on
Traffic structure but be careful with the size of the dataset as it does not
scale well:

.. jupyter-execute::

    from traffic.data.samples import quickstart

    subset = quickstart[["TVF22LK", "EJU53MF", "TVF51HP", "TVF78YY", "VLG8030"]]
    subset = subset.resample("10s").eval()
    assert subset is not None

    subset.map_leaflet(zoom=8)
