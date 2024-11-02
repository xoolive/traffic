Tutorial sessions
=================

I have been running several tutorial sessions where I usually use data recorded
from a flight I took to the tutorial venue place in order to illustrate the
possibilities of the traffic library.  A regular tutorial session is rather
interactive, but the demonstration usually goes along showcasing the following
possibilities.


I usually record data with `jet1090 <https://lib.rs/crates/jet1090>`_ and
traffic can parse the resulting JSONL file.

.. jupyter-execute::

    from traffic.core import Traffic

    t = Traffic.from_file("tutorial.jsonl.gz")
    t

We see several aircraft in the traffic structures, but want to only select the
``BAW3AK`` trajectory. In :class:`~traffic.core.Traffic` structures, we can
select trajectories based on the ``icao24`` address (the aircraft identifier) or
on the ``callsign``. However, only ``icao24`` fields are present in all
messages. It is important to reassign a callsign to the selected trajectory in
order to propagate its value along all lines.

.. jupyter-execute::

    f = t['400f99'].assign(callsign="BAW3AK")
    f

A :class:`~traffic.core.Flight` appears with a specific representation in IPython
or Jupyter like environments. From that point, we can access several methods. It
is often comfortable to start by resampling the trajectory and get proper state
vectors:

.. jupyter-execute::

    g = f.resample('1s')
    g

We can check the underlying pandas DataFrame:

.. jupyter-execute::

    g.data

.. jupyter-execute::

    g.start

.. jupyter-execute::

    g.callsign

.. jupyter-execute::

    g.duration

.. jupyter-execute::

    g.first('10 min')

A convenient way to explore trajectories is to have them displayed in a
:ref:`Leaflet <visualize trajectories with leaflet>` widget.

.. jupyter-execute::

    g.first('10 min').map_leaflet(zoom=14)

A common use case for on ground trajectories is to identify on which parking
positions an aircraft is parked. For that purpose, traffic uses data from the
OpenStreetMap database.

.. jupyter-execute::

    from traffic.data import airports

    airports['LFBO']

Representations are available for matplotlib, altair or more frameworks. The
altair representation is accessible and configurable here:

.. jupyter-execute::

    airports['LFBO'].geoencode()

Any attribute that is a possible value for the aeroway key on OpenStreetMap can
be accessed on the airport structure, resulting in a GeoDataFrame that we
further use for our analysis.

.. jupyter-execute::

    airports['LFBO'].parking_position

Next, the method :meth:`~traffic.core.Flight.on_parking_position` selects all
segments that intersect any parking positions on the airport. Since a trajectory
can intersect a geometry several times, the resulting structure is a
:class:`~traffic.core.FlightIterator`. If we want a :class:`~traffic.core.Flight`,
we need to select one of the segments in the iterator:

.. jupyter-execute::

    g.first('10 min').on_parking_position('LFBO')


.. jupyter-execute::

    # Meaning: the longest option in duration
    g.first('10 min').on_parking_position('LFBO').max('duration')


This method is used in the implementation of the
:meth:`~traffic.core.Flight.pushback` detection method.

.. jupyter-execute::

    g.first('10 min').pushback('LFBO')

We can highlight this part in the Leaflet widget:

.. jupyter-execute::

    g.first('10 min').map_leaflet(
        zoom=16,
        highlight={"red": 'pushback("LFBO")'},
    )

We can also look at the last part of the trajectory:

.. jupyter-execute::

    g.last('30 min').map_leaflet(zoom=10)

There is a :meth:`~traffic.core.Flight.holding_pattern` detection method
included, but here the shape of the trajectory is not consistent with the usual
horse racetrack shape and is not labelled as is:

.. jupyter-execute::

    g.holding_pattern()

Convenient methods include the detection of landing runways (ILS means
“Instrument Landing System”, i.e., the system that guides aircraft on autopilot
to align perfectly with a runway). Check the documentation for the
:meth:`~traffic.core.Flight.aligned_on_ils` method.

.. jupyter-execute::

    g.aligned_on_ils('EGLL')

The :meth:`~traffic.core.FlightIterator.next()` method selects the first segment
in the list. We can then detect the actual landing time, or more precisely, use
the time at runway threshold as a proxy for the actual landing time.

.. jupyter-execute::

    g.aligned_on_ils('EGLL').next().stop

This can also be rewritten for legibility as:

.. jupyter-execute::

    g.next("aligned_on_ils('EGLL')").stop

The tentative holding pattern here in London Heathrow Airport is labelled as OCK
(stands for “Ockham”, colocated with the OCK VOR). VOR and other navaids are
listed in part in the ``data`` part of the library.


.. jupyter-execute::

    from traffic.data import navaids

    m = g.last('30 min').map_leaflet(
        zoom=10,
        highlight={
            "red": 'aligned_on_ils("EGLL")',
            "#f58518": 'aligned_on_navpoint("OCK")',
        },
    )
    m.add(navaids['OCK'])
    m

Other parts of the library focus on fuel estimation provided by `OpenAP
<https://openap.dev>`_. The library is fully integrated in traffic, and can be
used to estimate flight phases and emissions.

.. jupyter-execute::

    # resample first to limit the size of the javascript
    g.phases().resample('10s').chart('phase').encode(y="phase", color='phase').mark_point()

We can also estimate the fuel flow (and plot with Matplotlib):

.. jupyter-execute::

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True)
    g.plot_time(ax[0], y='altitude')

    takeoff_time = g.next("takeoff_from_runway('LFBO')").stop
    landing_time = g.next("aligned_on_ils('EGLL')").stop
    g.between(takeoff_time, landing_time).emission().plot_time(ax[1], y='fuelflow')

    for ax_ in ax:
        ax_.spines['right'].set_visible(False)
        ax_.spines['top'].set_visible(False)

We can also estimate the total fuel consumed:

.. jupyter-execute::

    g.between(takeoff_time, landing_time).emission().fuel_max

.. warning::

    Note that it is not reasonable to consider these models on the ground and that it can result to big discrepancies.

    .. jupyter-execute::

        g.emission().fuel_max

.. hint::

    For any column in the pandas DataFrame wrapped by a Flight structure,
    traffic provides attributes to aggregate all values in the column.
    ``g.fuel_max`` is equivalent to ``g.data.fuel.max()``.


Advanced topics
---------------

.. toctree::

    gallery/savan
    navigation/go_around
