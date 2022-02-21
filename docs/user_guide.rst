User guide
==========

The user guide covers all of traffic by topic area.

Each of the entries introduces a topic (such as data sources, flight trajectory
labelling, trajectory generation, statistical analysis, etc.) and discusses how
traffic approaches the problem with as many examples as possible.

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
   data_sources/opensky_impala

.. toctree::
   :maxdepth: 1

   data_sources/eurocontrol
   data_sources/export


Navigation events
-----------------

.. toctree::
   :maxdepth: 1

   tutorial/flight_phases
   tutorial/go_around
   tutorial/runway_changes
   tutorial/top_of_climb
   tutorial/flight_plan
   tutorial/fuel_burn
   tutorial/occupancy
   airac_usage


Beyond the library
------------------

.. toctree::
   :maxdepth: 1

   user_guide/own_data
   user_guide/plugins

Standalone applications
-----------------------

.. toctree::
   :maxdepth: 1

   user_guide/gui
   user_guide/docker

Frequently asked questions
--------------------------

.. toctree::
   :maxdepth: 1

   troubleshooting/installation
   troubleshooting/network
   troubleshooting/data_access
