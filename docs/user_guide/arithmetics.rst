How to use arithmetic operators on trajectories?
================================================

- :ref:`Visualization with the | (or) operator`
- :ref:`Concatenation with the + (plus) operator`
- :ref:`Differences on trajectories with the - (minus) operator`
- :ref:`Differences on collections with the - (minus) operator`
- :ref:`Indexation with the [] (bracket) operator`
- :ref:`Overlapping of trajectories with the & (and) operator`
- :ref:`Intersection of collections with the & (and) operator`

Visualization with the | (or) operator
--------------------------------------

The ``|`` (prononce `"or"`) operator is very useful in interactive environments
(IPython, Jupyter, etc.) in order to put data structures next to each other.
The result of this operation is not designed to be stored in any variable.

For example, we can preview two trajectories on the same line:

.. jupyter-execute::

    from traffic.data.samples import belevingsvlucht

    belevingsvlucht | belevingsvlucht.next("aligned_on_ils('EHAM')")


Concatenation with the + (plus) operator
----------------------------------------

The ``+`` operator is a simple concatenation operator. In other words, all the
pandas DataFrames of the given structures are concatenated and wrapped in a
Traffic object.

.. automethod:: traffic.core.Flight.__add__

.. automethod:: traffic.core.Traffic.__add__

With the same trajectories as above, we can construct a Traffic object:

.. jupyter-execute::

    from traffic.data.samples import belevingsvlucht, pixair_toulouse

    belevingsvlucht + pixair_toulouse

The ``sum`` built-in function can be used to concatenate many flights into a
Traffic collection.


Differences on trajectories with the - (minus) operator
-------------------------------------------------------

The ``-`` operator serves a way to remove segments in a trajectory. The
resulting structure will consist in 0, 1 or many trajectory segments wrapped in
a FlightIterator object.

.. automethod:: traffic.core.Flight.__sub__


For example, on the following sample trajectory, the aircraft performs many
landing attempts at Lelystad airport (EHLE) in the Netherlands, which are easily
labelled as go-arounds. The difference operator will result in the trajectory
section before and after all the landing attempts.

.. jupyter-execute::

    belevingsvlucht - belevingsvlucht.go_around("EHLE")

Differences on collections with the - (minus) operator
------------------------------------------------------

The ``-`` operators also serves as a way to remove flights from a Traffic
collection. The operator behaves slightly differently if both collections are
equipped with a ``flight_id`` attribute or not.

.. automethod:: traffic.core.Traffic.__sub__

For example, in the following sample dataset, we build a sub dataset of all
trajectories going through a navaid point called "ODINA" at the border between
Switzerland and Italy. The ``-`` operator produces a new dataset of
trajectories not going through ODINA.

.. jupyter-execute::
    :hide-output:

    from traffic.data.samples import switzerland

    through_odina = switzerland.has('aligned_on_navpoint("ODINA")').eval()
    difference = switzerland - through_odina

.. jupyter-execute::

    through_odina | difference


Indexation with the [] (bracket) operator
-----------------------------------------

.. automethod:: traffic.core.Flight.__getitem__

The indexation of a Traffic collection is intended to be used in the most
versatile way.  Any object that could identify a particular trajectory can be
used as a key.

.. automethod:: traffic.core.Traffic.__getitem__

The following lets us get one or many trajectories:

.. jupyter-execute::
    :hide-output:

    # The first trajectory in the dataset
    switzerland[0]
    # The ten first trajectories in the dataset
    switzerland[:10]
    # The trajectory assigned with callsign ``EZY12VJ``
    switzerland['EZY12VJ']

The DataFrame indexation can be useful for example with the following use case.
We want to get full trajectories of aircraft entering the Swiss airspace at a
particular hour.  We first compute the statistics for each trajectory, then
build a trajectory subcollection.


.. jupyter-execute::

    from traffic.data import eurofirs

    stats = (
        switzerland.clip(eurofirs["LSAS"])
        .summary(["icao24", "callsign", "start", "stop"])
        .eval(max_workers=2)
    )
    stats

.. jupyter-execute::

    # All trajectories quitting the airspace between 15:30 and 15:40
    subset = stats.query('stop.dt.floor("10 min") == "2018-08-01 15:30Z"')
    subset_1530 = switzerland[subset]  # subset is a pd.DataFrame
    subset_1530

The ``start`` and ``stop`` timestamps are based on the clipping of the
trajectories within the LSAS FIR boundaries. If we want to select the subset of
the original trajectories matching those (shorter) selected in ``subset_1530``,
we can use the bracket operator again:

.. jupyter-execute::

    # We might prefer the full trajectories that have been matched in subset_1530
    switzerland[subset_1530]  # subset is a Traffic object



Overlapping of trajectories with the & (and) operator
-----------------------------------------------------

When applied on Flight structures, the & operator expresses the concurrency of
two trajectories, i.e. the piece of trajectory that is flown while another
aircraft is flying.

.. automethod:: traffic.core.Flight.__and__

.. jupyter-execute::

    from traffic.data.samples import dreamliner_airfrance

    # expansion of the collection into two flights
    video, dreamliner = dreamliner_airfrance
    # the operator is not commutative
    video & dreamliner | dreamliner & video

Intersection of collections with the & (and) operator
-----------------------------------------------------

When applied on collections, the & operator returns the subset of trajectories
that are present in both collections.

.. automethod:: traffic.core.Traffic.__and__

.. jupyter-execute::

    # All trajectories entering or quitting the airspace between 15:30 and 15:40 through ODINA
    result = through_odina & switzerland[subset_1530]
    result

.. jupyter-execute::

    from traffic.data import navaids

    ODINA = navaids['ODINA']
    m = result.map_leaflet(
        center=ODINA.latlon,
        highlight=dict(red="aligned_on_navpoint('ODINA')")
    )
    m.add(ODINA)
    m
