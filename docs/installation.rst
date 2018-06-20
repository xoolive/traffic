Installation
============

The traffic library makes an intensive use of `pandas <https://pandas.pydata.org/>`_ DataFrames and of the `shapely <https://shapely.readthedocs.io/en/latest/>`_ GIS library.

NumPy/SciPy handles most of the computing but some intensive treatments require
the `cython <http://cython.org/>`_ framework for a native compilation.

The library relies on `requests <http://docs.python-requests.org/en/master/>`_
for calls to REST APIs, and on `maya <https://github.com/kennethreitz/maya>`_ by
the same Kenneth Reitz for parsing dates written in a human-friendly format.
`paramiko <http://www.paramiko.org/>`_ implements the SSH protocol in Pure
Python, giving access to SSH connection independently of the operating system.

The `cartotools <https://github.com/xoolive/cartotools>`_ library is also
recommended for an access to some more projections and for getting data from
OpenStreetMap. The great `tqdm <https://github.com/tqdm/tqdm>`_ provides the
progress bars.

We recommend cloning the latest version from the repository before installing it.

.. parsed-literal::
    python setup.py install

If you are not comfortable with that option, you can install the latest release:

.. parsed-literal::
    pip install traffic
