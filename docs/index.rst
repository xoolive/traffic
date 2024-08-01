
.. container:: title-image

   .. image:: _static/logo/logo_full.png
      :alt: traffic -- air traffic data processing with Python

Source code on `github <https://github.com/xoolive/traffic>`_

The traffic library helps to work with common sources of air traffic data.

Its main purpose is to provide data analysis methods commonly applied to
trajectories and airspaces. When a specific function is not provided,
the access to the underlying structure is direct, through an attribute pointing
to a pandas dataframe.

The library also offers facilities to parse and/or access traffic data from open
sources of ADS-B traffic like the `OpenSky Network
<https://opensky-network.org/>`_ or Eurocontrol DDR files. It is designed to be
easily extendable to other sources of data.

Contents
========

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   tutorial
   user_guide
   gallery
   api_reference
   publications
