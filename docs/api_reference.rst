API reference
=============

The `traffic` library is based on three main core classes for handling:

- aircraft trajectories are accessible through :class:`traffic.core.Flight`;
- collections of aircraft trajectories are :class:`traffic.core.Traffic`;
- airspaces and sectors are represented with :class:`traffic.core.Airspace`.

:class:`~traffic.core.Flight` and :class:`~traffic.core.Traffic` are wrappers
around :class:`pandas.DataFrame` with relevant efficiently implemented methods
for trajectory analysis. Airspaces take advantage of `shapely Geometries
<https://shapely.readthedocs.io>`_ for geometrical analysis.

:class:`~traffic.core.FlightIterator` have been introduced in order to
deal with iteration over pieces of trajectories. This structure allows for more
flexibility and accuracy when trying to identify specific events which may
happen several times along a trajectory.

:ref:`Lazy iteration <traffic.core.lazy>` has been implemented on top of
:class:`~traffic.core.Traffic` structures in order to chain operations
to be applied on every :class:`~traffic.core.Flight`  in the collection
and stack them before triggering only one iteration.

.. toctree::
   :maxdepth: 1

   api_reference/traffic.core.airspace
   api_reference/traffic.core.cache
   api_reference/traffic.core.distance
   api_reference/traffic.core.flight
   api_reference/traffic.core.flightplan
   api_reference/traffic.core.intervals
   api_reference/traffic.core.iterator
   api_reference/traffic.core.lazy
   api_reference/traffic.core.mixins
   api_reference/traffic.core.structure
   api_reference/traffic.core.sv
   api_reference/traffic.core.time
   api_reference/traffic.core.traffic
   api_reference/traffic.algorithms.filters
