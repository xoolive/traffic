How to access sample trajectories?
==================================

A bundle of sample trajectories covering various situations has been included in
the library for reference, testing and for providing a baseline to compare the
performance of various methods and algorithms.

All sample trajectories are available in the `traffic.data.samples` module. The
import automatically dispatch to :class:`~traffic.core.Flight` or
:class:`~traffic.core.Traffic` according to the nature of the data.

.. note::

    A subset of the sample trajectories are presented on this page. Other parts
    of the documentation (e.g. :ref:`calibration flights <calibration flights>`
    or :ref:`trajectory clustering <How to implement trajectory clustering?>`)
    may refer to other available sample trajectories.
    
Belevingsvlucht
~~~~~~~~~~~~~~~

https://www.belevingsvlucht.nl/

On May 30th 2018, test flights were conducted along future routes arriving and
departing from Lelystad Airport in the Netherlands. The purpose of this flight
operated by a Boeing 737 owned by Transavia was to assess noise exposure in
neighbouring cities.

.. jupyter-execute::

    from traffic.data.samples import belevingsvlucht
    belevingsvlucht


Dreamliner Air France
~~~~~~~~~~~~~~~~~~~~~

After getting their first Boeing 787 Dreamliner, Air France equipped a
Socata TBM 900 and shot the following video in 8k. Both trajectories are
(partly) available as sample flights.

.. raw:: html

    <iframe max-width="800" width="100%" height="400" style="margin-bottom: 1em;"
     src="https://www.youtube.com/embed/JkDsYkhCl-U?rel=0&amp;controls=0&amp;showinfo=0"
     frameborder="0" allowfullscreen></iframe>

You can explore how the two trajectories intertwine for shooting purposes:

.. jupyter-execute::

    from traffic.data.samples import dreamliner_airfrance
    dreamliner_airfrance

.. jupyter-execute::

    dreamliner_airfrance['AFR787V'] | dreamliner_airfrance['FWKDL']

.. jupyter-execute::

    dreamliner_airfrance.map_leaflet(
        highlight=dict(red=lambda f: f.query('icao24 == "3900fb"')),
        center=(43.5, 3.37),
        zoom=9,
    )

The following `Altair <https://altair-viz.github.io/>`_ specification lets you
plot the altitude profile (clamped to 20,000 ft) for the two trajectories. The
steps in the altitude profile are due to the low sampling rate in the data:

.. jupyter-execute::

    import altair as alt

    chart = (
        alt.layer(
            *list(
                flight.chart().encode(
                    alt.X("utchoursminutes(timestamp)", title=""),
                    alt.Y(
                        "altitude",
                        title="",
                        scale=alt.Scale(domain=(5000, 20000), clamp=True),
                    ),
                    alt.Color("callsign", title="Callsign of the aircraft"),
                )
                for flight in dreamliner_airfrance.between(
                    "2017-12-01 14:40", "2017-12-01 15:40"
                )
            )
        )
        .properties(title="Altitude (in ft)", width=600)
        .configure_title(anchor="start", font="Lato", fontSize=15, dy=-5)
        .configure_axis(labelFontSize=12)
        .configure_legend(orient="bottom", titleFont="Lato", titleFontSize=13)
    )
    chart



Airbus tree
~~~~~~~~~~~

Before Christmas 2017, an Airbus pilot in Germany has delivered an early
festive present by tracing the outline of an enormous Christmas tree
during a test flight.

.. jupyter-execute::

    from traffic.data.samples import airbus_tree
    airbus_tree

Other trajectories
~~~~~~~~~~~~~~~~~~

Even though all trajectories are accessible from the ``traffic.data.samples``
module, they are in practice organised by categories:

.. jupyter-execute::

    from pkgutil import iter_modules  # one of Python inspection modules
    from traffic.data import samples

    list(category.name for category in iter_modules(samples.__path__))

For instance, National Geographic Institutes sometimes conduct aerial surveys
including photography and LIDAR measurements from aircraft, some of such
trajectories are available in the ``surveys`` module (category).

In each category, you can list all available trajectories:

.. jupyter-execute::

    from traffic.data.samples import surveys
    surveys.__all__

Then the same trajectory is available both from the module of each category (here ``surveys``) and
from the root ``samples`` module:

.. jupyter-execute::

    surveys.pixair_toulouse | samples.pixair_toulouse

So in practice, both imports are valid (and completion helps in both cases):

.. jupyter-execute::

    from traffic.data.samples import pixair_toulouse
    from traffic.data.samples.surveys import pixair_toulouse

You can make a Traffic object from a category module:

.. jupyter-execute::

    from traffic.core import Traffic

    t_surveys = Traffic.from_flights(  # actually, this is equivalent to sum(...)
        getattr(surveys, name)  # getattr(surveys, "pixair_toulouse") is surveys.pixair_toulouse
        .assign(flight_id=name)  # gives a flight_id to each trajectory
        for name in surveys.__all__
    )
    t_surveys

As illustrated throughout this documentation, the ``|`` operator (``or_``)
concatenates Jupyter representations of eligible objects. Therefore, the
following trick lets you apply the ``or_`` operator to all flights available in
the ``surveys`` category:

.. jupyter-execute::

    from functools import reduce
    from operator import or_

    # this will do surveys.flight1 | surveys.flight2 | surveys.flight3 | etc.
    reduce(or_, t_surveys)


