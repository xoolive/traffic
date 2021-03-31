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
`cartes <https://github.com/xoolive/cartes>`_ library, which leverages
access to more projections and to data from OpenStreetMap. More dynamic
visualisations in Jupyter Lab are accessible thanks to the `altair <https://altair-viz.github.io/>`_ and `ipyleaflet
<http://ipyleaflet.readthedocs.io/>`_ libraries; other exports to various formats
(including CesiumJS or Google Earth) are also available.

We recommend cloning the latest version from the repository before installing
it.

.. parsed-literal::
    git clone https://github.com/xoolive/traffic
    cd traffic/
    pip install .

If you are not comfortable with that option, you can always install the latest
release:

.. parsed-literal::
    pip install traffic

.. warning::
    `cartopy` and `shapely` have strong dependencies to dynamic libraries which
    may not be available on your system by default. If possible, install
    `Anaconda <https://www.anaconda.com/distribution/#download-section>`_, 
    then create a virtualenv and run the next commands in the prompt/terminal:

    .. parsed-literal::
       conda install cartopy shapely
       # then _either_ with pip (stable version)
       pip install traffic
       # _or_ from sources (dev version)
       pip install .

.. toctree::
   :maxdepth: 1
   :caption: Frequently Asked Questions

   troubleshooting
   docker