How to access ADS-B data from OpenSky live API?
===============================================

.. warning::

  OpenSky data are subject to particular `terms of use
  <https://opensky-network.org/about/terms-of-use>`_. In particular, if you plan
  to use data for commercial purposes, you should `contact them
  <https://opensky-network.org/about/contact>`_.

Anonymous access to the OpenSky live API is possible, but functionalities may be
limited. The first thing to do once you have an account is to put your
credentials in you configuration file. Add the following lines to the [opensky]
section of your configuration file.

.. parsed-literal::

    [opensky]
    username =
    password =

You can check the path to your configuration file here. The path is
different according to OS versions so do not assume anything and check
the contents of the variable.

.. code:: python

    >>> import traffic
    >>> traffic.config_file
    PosixPath('/home/xo/.config/traffic/traffic.conf')


State vectors
-------------

The most basic usage for the OpenSky REST API is to get the instant
position for all aircraft. This part actually does not require
authentication.

.. jupyter-execute::
  :raises:

    import matplotlib.pyplot as plt
    import pandas as pd

    from cartes.crs import EuroPP
    from cartes.utils.features import countries

    from traffic.data import opensky

    sv = opensky.api_states()

    with plt.style.context('traffic'):
        fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))

        ax.add_feature(countries())
        ax.gridlines()
        ax.set_extent((-7, 15, 40, 55))
        ax.spines['geo'].set_visible(False)

        sv.plot(ax, s=10, color="#4c78a8")

        now = pd.Timestamp("now", tz="utc")
        ax.set_title(
          f"Snapshot generated at {now:%Y-%m-%d %H:%MZ}",
          fontsize=14
        )


Flight tables
-------------

Flight tables are accessible by airport (use the ICAO code) given temporal
bounds:

.. jupyter-execute::
  :raises:

  # Have you seen Santa Claus coming to Toulouse? 
  opensky.api_arrival("LFBO", "2021-12-24 20:00", "2021-12-25 06:00")


.. jupyter-execute::
  :raises:

  # Or maybe leaving?
  opensky.api_departure("LFBO", "2021-12-24 20:00", "2021-12-25 06:00")

A basic route database is also accessible through the REST API:

.. jupyter-execute::
  :raises:

  opensky.api_routes("AFR292")
