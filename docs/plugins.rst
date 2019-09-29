.. raw:: html
    :file: ./embed_widgets/plugins.html

Plugins
=======

We can export traffic structures to different visualisation tools. 
We use here the data produced on the `Quickstart </quickstart.html>`_ page.

.. code:: python

    from traffic.data.samples import quickstart
    
    def landing_trajectory(flight: "Flight") -> bool:
        return (
            flight.min("altitude") < 10_000 and
            flight.mean("vertical_rate") < -500
        )

    demo = (
        quickstart
        # non intersecting flights are discarded
        .intersects(lfbo_tma)
        # intersecting flights are filtered
        .filter()
        # filtered flights not matching the condition are discarded
        .filter_if(landing_trajectory)
        # stay below 25000ft
        .query('altitude < 25000')
        # final multiprocessed evaluation (4 cores) through one iteration
        .eval(max_workers=4)
    )


Leaflet
~~~~~~~

`Leaflet <http://leafletjs.com/>`__ offer a Python `widget
<https://github.com/jupyter-widgets/ipyleaflet>`__ for Jupyter Lab. Flights and
Airspaces can easily be plotted into such widgets. The Traffic extracted above
can be conveniently explored in the following widget.

Just for fun, you can zoom up to the airport level and check the runway used for
landing.

.. code:: python

    from ipyleaflet import Map, basemaps
    from ipywidgets import Layout

    map_ = Map(
        center=(43.5, 1.5),
        zoom=7,
        basemap=basemaps.Stamen.Terrain,
        layout=Layout(width="100%", max_width="800px", height="500px"),
    )

    map_.add_layer(nm_airspaces["LFBOTMA"])
    for flight in demo:
        map_.add_layer(flight, color="#990000", weight=2)

    map_

.. raw:: html

    <script type="application/vnd.jupyter.widget-view+json">
    {
        "version_major": 2,
        "version_minor": 0,
        "model_id": "0f287b37251c4aa28883bf0b3daa695b"
    }
    </script>

CesiumJS
~~~~~~~~

`CesiumJS <http://cesiumjs.org/>`__ is a great tool for displaying and
animating geospatial 3D data. The library provides an export of a
Traffic structure to a czml file. A copy of this file is available in
the ``data/`` directory. You may drag and drop it on the
http://cesiumjs.org/ page after you open it on your browser.

.. code:: python

    demo.to_czml('data/sample_cesium.czml')