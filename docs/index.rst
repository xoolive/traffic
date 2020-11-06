.. traffic documentation master file, created by
   sphinx-quickstart on Mon Jun 18 22:56:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

traffic – Air traffic data processing in Python
===============================================

Source code on `github <https://github.com/xoolive/traffic>`_

The traffic library helps working with common sources of air traffic data.

Its main purpose is to provide data analysis methods commonly applied to
trajectories and airspaces. When a specific function is not provided,
the access to the underlying structure is direct, through an attribute pointing
to a pandas dataframe.

The library also offers facilities to parse and/or access traffic data from open
sources of ADS-B traffic like the `OpenSky Network
<https://opensky-network.org/>`_ or Eurocontrol DDR files. It is designed to be
easily extendable to other sources of data.



.. toctree::
   :maxdepth: 1

   installation
   quickstart
   core_structure
   data
   algorithms
   export
   advanced
   gallery
   scenarios
   publications

