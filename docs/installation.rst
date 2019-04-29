Installation
============

The traffic library makes an intensive use of `pandas
<https://pandas.pydata.org/>`_ DataFrames and of the `shapely
<https://shapely.readthedocs.io/en/latest/>`_ GIS library.

The library relies on `requests <http://docs.python-requests.org/en/master/>`_
for calls to REST APIs. `paramiko <http://www.paramiko.org/>`_ implements the
SSH protocol in Pure Python, giving access to SSH connection independently of
the operating system.

Static visualisation tools are accessible with Matplotlib through the
`cartotools <https://github.com/xoolive/cartotools>`_ library, which leverages
access to more projections and to data from OpenStreetMap. More dynamic
visualisations in Jupyter Lab are accessible thanks to the `altair <https://altair-viz.github.io/>`_ and `ipyleaflet
<http://ipyleaflet.readthedocs.io/>`_ libraries; other exports to various formats
(including CesiumJS or Google Earth) are also available.

We recommend cloning the latest version from the repository before installing it.

.. parsed-literal::
    python setup.py install

If you are not comfortable with that option, you can install the latest release:

.. parsed-literal::
    pip install traffic

Warning
-------

`cartopy` and `shapely` have strong dependencies to dynamic libraries which
may not be available on your system by default.

Before reporting an issue, please try to use an Anaconda environment. Other
installations may work but the Anaconda way proved to cause much less issues.

.. parsed-literal::
   conda install cartopy shapely
