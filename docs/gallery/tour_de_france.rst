Tour de France 2020
===================


During the Tour de France, aircraft are contracted to `relay TV signals
<https://arstechnica.com/cars/2019/07/how-helicopters-bring-us-amazing-views-of-the-tour-de-france/>`_
from helicopters filming the riders. See how their trajectories fits the
official map at the bottom of the page.

.. jupyter-execute::
    :code-below:

    from traffic.data.samples import tour_de_france
    from ipywidgets import Layout

    def straight_ratio(flight) -> float:
        return flight.distance() / flight.cumulative_distance(False, False).max("cumdist")

    preprocessed = (
        tour_de_france.iterate_lazy(iterate_kw=dict(by="1h"))
        .assign_id()
        .apply_time("5 min", straight_ratio=straight_ratio)
        .query("straight_ratio < .5")
        .max_split()
        .longer_than("1h")
        .eval()
    )

    stats = preprocessed.summary(["flight_id", "start", "stop", "duration"]).eval()

    m = preprocessed.map_leaflet(
        zoom=6, layout=Layout(max_width="600px", height="600px"),
    )

    display(m)
    display(stats)


.. image:: ../_static/tour_de_france_2020.jpg
   :scale: 60%
   :alt: Tour de France 2020
   :align: left
