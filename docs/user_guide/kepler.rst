.. raw:: html
    :file: ./embed_widgets/kepler.html


Kepler.gl
~~~~~~~~~

`Kepler.gl <http://www.kepler.gl>`__ is a data-agnostic,
high-performance web-based application for visual exploration
of large-scale geolocation data sets. It is built on top of
Mapbox GL and deck.gl, and can render millions of points
representing thousands of trips and perform spatial aggregations
on the fly. A Jupyter binding is available `here
<https://github.com/keplergl/kepler.gl>`__: Flights, Traffic, 
Airspaces, and other data can easily be plotted into the widget.


.. warning:: 

    The plugin must be `activated <plugins.html>`__ in your configuration file.

The following example displays the same example as with Leaflet,
together with extra data: German airports, VORs in Ireland and French
FIRs.

.. code:: python

    from traffic.data import airports, navaids, eurofirs

    from keplergl import KeplerGl

    map_ = KeplerGl(height=500)

    # add a Flight or a Traffic
    map_.add_data(belevingsvlucht, name="Belevingsvlucht")
    map_.add_data(demo, name="Quickstart trajectories")
    
    # also other sources of data
    map_.add_data(  # airports (a subset)
        airports.query('country == "Germany"'),
        name="German airports"
    )
    map_.add_data(  # navaids (a subset)
        navaids.extent("Ireland").query("type =='VOR'"),
        name="Irish VORs"
    )
    map_.add_data(lfbo_tma, "Toulouse airport TMA")

    # you can write your own generator to send data to the widget
    map_.add_data(
        (
            fir for name, fir in eurofirs.items() 
            if name.startswith("LF")
        ),
        "French FIRs"
    )

    map_

.. raw:: html

    <script type="application/vnd.jupyter.widget-view+json">
    {
        "version_major": 2,
        "version_minor": 0,
        "model_id": "21422b15233c41dfaef0616eb0f97d4e"
    }
    </script>
