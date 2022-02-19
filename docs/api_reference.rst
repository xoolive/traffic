API reference
=============

The `traffic` library is based on three main core classes for handling:

- aircraft trajectories are accessible through
  `traffic.core.Flight <traffic.core.flight.html>`_;
- collections of aircraft trajectories are
  `traffic.core.Traffic <traffic.core.traffic.html>`_;
- airspaces and sectors are represented with
  `traffic.core.Airspace <traffic.core.airspace.html>`_.

`Flight <traffic.core.flight.html>`_ and `Traffic <traffic.core.traffic.html>`_
are wrappers around `pandas DataFrames <https://pandas.pydata.org/>`_ with
relevant efficiently implemented methods for trajectory analysis. Airspaces take
advantage of `shapely Geometries <https://shapely.readthedocs.io>`_ for
geometrical analysis.

`FlightIterator <traffic.core.iterator.html>`_ has been introduced in order to
deal with iteration over pieces of trajectories. This structure allows for more
flexibility and accuracy when trying to identify specific events which may
happen several times along a trajectory.

`Lazy iteration <traffic.core.lazy.html>`_ has been implemented on top of
`Traffic <traffic.core.traffic.html>`_ structures in order to chain operations
to be applied on every `Flight <traffic.core.flight.html>`_ in the collection
and stack them before triggering only one iteration.

Lastly, the `algorithm` module contains advanced implementations with methods
being monkey-patched on `Flight <traffic.core.flight.html>`_ and `Traffic
<traffic.core.traffic.html>`_ structures. Currently, they cover:

- `navigation events <navigation.html>`_ (e.g., holding patterns, go-arounds),
- `trajectory clustering <clustering.html>`_,
- `trajection generation <generation.html>`_ and
- `closest point of approach <cpa.html>`_.


.. toctree::
   :hidden:

   api_reference/traffic.core.flight
   api_reference/traffic.core.traffic
   api_reference/traffic.core.airspace
   api_reference/traffic.core.iterator
   api_reference/traffic.core.lazy
   navigation
   clustering
   generation
   cpa
