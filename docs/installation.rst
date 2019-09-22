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

We recommend cloning the latest version from the repository before installing
it.

.. parsed-literal::
    python setup.py install

If you are not comfortable with that option, you can install the latest
release:

.. parsed-literal::
    pip install traffic

.. warning::
    `cartopy` and `shapely` have strong dependencies to dynamic libraries which
    may not be available on your system by default. If possible, install
    `Anaconda <https://www.anaconda.com/distribution/#download-section>`_, then:

    .. parsed-literal::
       conda install cartopy shapely
       # then _either_ with pip (stable version)
       pip install traffic
       # _or_ from sources (dev version)
       python setup.py install


Troubleshooting
---------------

The following reported problems are not problems with the traffic library per
se but with limitations set by dependencies.

Python crashes when I try to reproduce plots in the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(or, "My Jupyter kernel crashes...")

There must be something wrong with your Cartopy and/or shapely installation.
These libraries strongly depend on the ``geos`` and ``proj`` libraries. You
must have shapely and Cartopy versions matching the correct versions of these
libraries.

The problem is sometimes hard to understand, and you may end up fixing it
without really knowing how.

If you don't know how to install these dependencies, start with a **fresh**
Anaconda distribution and install the following libraries *the conda way*:

.. parsed-literal::
   conda install cartopy shapely

If it still does not work, try something along:

.. parsed-literal::
   conda uninstall cartopy shapely
   pip uninstall cartopy shapely
   # be sure to remove all previous versions before installing again
   conda install cartopy shapely

If it still does not work, try again with:

.. parsed-literal::
   conda uninstall cartopy shapely
   pip uninstall cartopy shapely
   # this forces the recompilation of the packages
   pip install --no-binary :all: cartopy shapely


Widgets do not display in Jupyter Lab or Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After executing a cell, you may see one of the following output:

.. parsed-literal::
    A Jupyter Widget
    # or
    Error displaying widget
    # or
    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))
    # or
    Map(basemap={'url': 'https://{s}.tile.openstreetmap.org/…

You will need to activate the widgets extensions:

- with Jupyter Lab:

    .. parsed-literal::
       jupyter labextension install @jupyter-widgets/jupyterlab-manager
       jupyter labextension install jupyter-leaflet

- with Jupyter Notebook:

    .. parsed-literal::
       jupyter nbextension enable --py --sys-prefix widgetsnbextension
       jupyter nbextension enable --py --sys-prefix ipyleaflet
