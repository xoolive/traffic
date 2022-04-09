.. raw:: html
    :file: ./embed_widgets/leaflet.html


Leaflet
~~~~~~~

`Leaflet <http://leafletjs.com/>`__ offers a Python `widget
<https://github.com/jupyter-widgets/ipyleaflet>`__ for Jupyter Lab. Flights and
Airspaces can easily be plotted into such widgets. The Traffic extracted above
can be conveniently explored in the following widget.

Just for fun, you can zoom up to the airport level and check the runway used for
landing.

.. warning:: 

    The plugin must be `activated <plugins.html>`__ in your configuration file.

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
