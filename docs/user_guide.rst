User guide
==========

The user guide covers all of ``traffic`` by topic area.

Each of the entries introduces a topic (such as data sources, flight trajectory
labelling, trajectory generation, statistical analysis, etc.) and discusses how
``traffic`` approaches the problem with as many examples as possible.

.. tip::

   | There are obviously as many possible use cases as you can imagine for the
     library.
   | **Contributions are open** in this part of the documentation. Do consider
     sharing your experience with the library where it is the less documented.


Sources of data
---------------

.. toctree::
   :maxdepth: 1

   data_sources/samples
   data_sources/airports
   data_sources/aircraft
   data_sources/navigation
   data_sources/airspaces
   data_sources/flightplans
   data_sources/opensky_rest
   data_sources/opensky_db
   data_sources/decode

.. toctree::
   :maxdepth: 1

   data_sources/eurocontrol
   data_sources/export

Good practices
--------------

.. toctree::
   :maxdepth: 1

   user_guide/arithmetics
   user_guide/projection
   user_guide/processing
   user_guide/simplify
   user_guide/own_data

Navigation events
-----------------

.. toctree::
   :maxdepth: 1

   tutorial/flight_phases
   airac_usage
   tutorial/go_around
   tutorial/runway_changes
   tutorial/top_of_climb
   tutorial/flight_plan
   tutorial/fuel_burn

Statistical analysis
--------------------

.. toctree::
   :maxdepth: 1

   tutorial/occupancy
   clustering
   tutorial/generation
   tutorial/cpa

Standalone applications
-----------------------

.. toctree::
   :maxdepth: 1

   user_guide/plugins
   user_guide/cli_tui
   user_guide/docker

Troubleshooting
---------------

.. toctree::
   :maxdepth: 1

   troubleshooting/installation
   troubleshooting/network
   troubleshooting/data_access
