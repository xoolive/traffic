Core structure
==============

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

**Contents**

.. toctree::
   :maxdepth: 1

   traffic.core.flight
   traffic.core.traffic
   traffic.core.airspace
