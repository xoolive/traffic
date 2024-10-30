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

Data visualisation
------------------

.. toctree::
   :maxdepth: 1

   visualize/projection

.. toctree::
   :maxdepth: 1

   visualize/leaflet
   visualize/plotly

Good practices
--------------

.. toctree::
   :maxdepth: 1

   user_guide/own_data
   user_guide/processing
   user_guide/simplify
   user_guide/arithmetics

Navigation events
-----------------

.. toctree::
   :maxdepth: 1

   navigation/flight_phases
   navigation/go_around
   navigation/runway_changes
   navigation/holding_pattern
   navigation/top_of_climb
   navigation/flight_plan
   navigation/fuel_burn

Statistical analysis
--------------------

.. toctree::
   :maxdepth: 1

   tutorial/occupancy
   clustering
   tutorial/generation
   tutorial/cpa

Troubleshooting
---------------

.. toctree::
   :maxdepth: 1

   troubleshooting/installation
   troubleshooting/docker
   troubleshooting/network
   troubleshooting/data_access
