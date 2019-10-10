Plugins
=======

Plugins are pieces of software which are designed to extend the 
basic functionalities of the traffic library. Plugins can be
implemented through a `registration <https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`__
mechanism and selectively activated in the configuration file.

Some plugins are provided by the traffic library with visualisation
facilities for Leaflet, Kepler.gl and CesiumJS.

Plugin activation
-----------------

You may activate plugins in the configuration file:


.. code:: python

    >>> import traffic
    >>> traffic.config_file
    PosixPath('/home/xo/.config/traffic/traffic.conf')

Then edit the following line according to the plugins you want to
activate:

::

    [plugins]
    enabled_plugins = Leaflet, Kepler, CesiumJS


Available plugins
-----------------

.. toctree::
   :maxdepth: 1

   leaflet
   kepler
   cesium
   bluesky
   


The examples are provided using the data produced in the
`Quickstart </quickstart.html>`_ page.

.. code:: python

    from traffic.data.samples import quickstart, lfbo_tma
    
    def landing_trajectory(flight: "Flight") -> bool:
        return (
            flight.min("altitude") < 10_000 and
            flight.mean("vertical_rate") < -500
        )

    demo = (
        quickstart
        # non intersecting flights are discarded
        .intersects(lfbo_tma)
        # intersecting flights are filtered
        .filter()
        # filtered flights not matching the condition are discarded
        .filter_if(landing_trajectory)
        # stay below 25000ft
        .query('altitude < 25000')
        # final multiprocessed evaluation (4 cores) through one iteration
        .eval(max_workers=4)
    )

Plugin registration
-------------------

You may write your own plugins to monkey-patch the library and
registrate them using entry points in your setup.py configuration.

Write your code in a ``my_traffic_plugin/plugin.py`` and use the
following sample setup.py to registrate your plugin. Then edit the
configuration file and add `MyPlugin` to the list of enabled plugins.

You may then ``python setup.py install`` your plugin. Put the
monkey-patching in an ``_onload()`` function that will be called iff the
plugin is enabled. (Check an example on the `github repository
<https://github.com/xoolive/traffic/blob/master/traffic/plugins/leaflet.py>`__)


.. code:: python

    from setuptools import setup

    setup(
        name="my_traffic_plugin",
        version="0.1",
        author="Antoine",
        author_email="saintex@aeropostale.fr",
        license="MIT",
        description="My first traffic plugin",
        entry_points={
            "traffic.plugins": [
                "MyPlugin = my_traffic_plugin.plugin"
            ]
        },
        install_requires=["traffic"]
    )
