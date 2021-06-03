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

We recommend creating a fresh conda environment for a first installation:

.. parsed-literal::
    conda create -n traffic -c conda-forge python=3.9 traffic
    conda activate traffic

You may as well install traffic in an existing environment:

.. parsed-literal::
    conda install -c conda-forge traffic

For the most recent development version:

.. parsed-literal::
    git clone https://github.com/xoolive/traffic
    cd traffic/
    pip install .

.. toctree::
   :maxdepth: 1
   :caption: Frequently Asked Questions

   troubleshooting
   docker