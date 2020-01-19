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

Installation
------------

We recommend cloning the latest version from the repository before installing
it.

.. parsed-literal::
    git clone https://github.com/xoolive/traffic
    cd traffic/
    python setup.py install

If you are not comfortable with that option, you can always install the latest
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

The following reported problems are not about the traffic library per
se. Some are related to limitations set by dependencies.

I can't access resources from the Internet as I am behind a proxy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All network accesses are made with the `requests
<https://requests.readthedocs.io/>`_ library (or the `paramiko
<http://www.paramiko.org/>`_ library for the Impala shell). Usually, if your
environment variables are properly set, you should not encounter any particular
proxy issue. However, there are always situations where it may help to manually
force the proxy settings in the configuration file.

Edit your configuration file (you may find where it is located in
traffic.config_file) and add the following section. Uncomment and edit options
according to your network configuration.

.. parsed-literal::
    ## This sections contains all the specific options necessary to fit your
    ## network configuration (including proxy and ProxyCommand settings)
    [network]

    ## input here the arguments you need to pass as is to requests
    # http.proxy = http://proxy.company:8080
    # https.proxy = http://proxy.company:8080
    # http.proxy = socks5h://localhost:1234
    # https.proxy = socks5h://localhost:1234

    ## input here the ProxyCommand you need to login to the Impala Shell
    ## WARNING:
    ##    Do not use %h and %p wildcards.
    ##    Write data.opensky-network.org and 2230 explicitly instead
    # ssh.proxycommand = ssh -W data.opensky-network.org:2230 proxy_ip:proxy_port
    # ssh.proxycommand = connect.exe -H proxy_ip:proxy_port data.opensky-network.org 2230


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
       jupyter labextension install keplergl-jupyter

- with Jupyter Notebook:

    .. parsed-literal::
       jupyter nbextension enable --py --sys-prefix widgetsnbextension
       jupyter nbextension enable --py --sys-prefix ipyleaflet
       jupyter nbextension enable --py --sys-prefix keplergl
