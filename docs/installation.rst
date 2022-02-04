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

Latest release
--------------

We recommend creating a fresh conda environment for a first installation:

.. parsed-literal::

    # Installation
    conda create -n traffic -c conda-forge python=3.9 traffic
    conda activate traffic

Adjust the Python version (>=3.7) and append packages you may need for future works (e.g. ``jupyterlab``, ``xarray``, etc.)

Then activate the environment each time you need to use the ``traffic`` library:

.. parsed-literal::

    conda activate traffic

.. warning::

    Please only report installation issues in fresh conda environments.

.. hint::

    Consider using `mamba <https://github.com/mamba-org/mamba>`_ for a faster Conda experience.

Updating traffic
----------------

.. parsed-literal::

    # -n option is followed by the name of the environment
    conda update -n traffic -c conda-forge traffic


Development version
-------------------

You may also install or update ``traffic`` in an existing environment with pip:

.. parsed-literal::

    pip install --upgrade traffic

For the most recent development version, clone the Github repository:

.. parsed-literal::

    git clone https://github.com/xoolive/traffic
    cd traffic/
    pip install .[dev]

If you intend to file a pull request, please activate ``pre-commit`` hooks:

.. parsed-literal::

    pre-commit install

.. toctree::
   :maxdepth: 1
   :caption: Frequently Asked Questions

   troubleshooting
   docker
