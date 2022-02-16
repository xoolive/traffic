User guide
==========

The user guide covers all of traffic by topic area. Each of the subsections
introduces a topic (such as flight trajectory labelling, trajectory generation,
statistical analysis, etc.) and discusses how traffic approaches the problem
with as many examples as possible.


Sources of data
---------------

.. toctree::
   :maxdepth: 1

   airports
   aircraft
   navaids
   airways
   samples
   firs
   opensky_rest
   opensky_impala
   airac_usage
   so6_usage
   b2b_usage
   export


Navigation events
-----------------

- How to find flight phases on a trajectory?
- How to select go-arounds from a set of trajectories?
- How to select runway changes from a set of trajectories?
- How to compute the top of climb/top of descent of a trajectory?
- How to infer a flight plan from a trajectory?
- How to estimate the fuel burnt by an aircraft?

Statistics
----------

- How to compute an occupancy graph?
- How to perform trajectory clustering?


Extend the library
------------------

- How to use ``traffic`` with your own data?

.. toctree::
   :maxdepth: 1

   plugins

Standalone applications
-----------------------

.. toctree::
   :maxdepth: 1

   gui
   docker

Troubleshooting
---------------

.. toctree::
   :maxdepth: 1

   troubleshooting