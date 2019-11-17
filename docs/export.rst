Exporting and storing data
==========================

The traffic library is based on the `pandas
<https://pandas.pydata.org/>`_ library for representing and manipulating trajectories and on the `shapely
<https://shapely.readthedocs.io/en/latest/>`_ library for manipulating geographical information like airways, beacons, airports and airspaces.

Various parsers are provided (feel free to file a PR if you use other tools
with different data sources) but after applying different operations to
imported data, you may want to export data for storing, sharing and loading
again later.

Traffic and Flight structures
-----------------------------

The question applies to pandas dataframes as well, with many opinions
available on the net. In general, the question boils down to whether you want
to distribute the data, how fast you need to access it and how long you need
to keep it.

- CSV (comma separated values) is a pretty standard and widely acknowledged
  format (modulo the definition of the separator). It is easy to parse but it
  can be slow when data gets large. Also it doesn’t contain information about
  types so you need to check dtypes and transform them manually if need be. A
  good rule of thumb could be to parse CSV data only once and to use another
  format for storing it for future use.

- JSON (JavaScript Object Notation) is another lightweight text notation,
  human readable, also slow to parse. The added value compared to CSV is that
  you can distinguish boolean, numerical and string values.

- `Pickle <https://docs.python.org/3/library/pickle.html>`_ is the standard
  format for Python serialisation. The binary representation of data is dumped
  as is in a file, no question asked. It is fast to read and write and you are
  sure to recover your data after you restart your Python interpreter/kernel.
  The downside is that the serialisation format may change with Python and
  pandas versions so it is not a good format for sharing and storing data in
  the time.

- HDF (Hierarchical Data Format) is a cross platform and cross language
  standard format for storing very large amounts of data. You may need extra
  dependencies to read and write from this format.

- `Apache Parquet <https://parquet.apache.org/>`_ is another columnar cross
  platform and cross language standard storage format. Its implementation
  inside pandas is very fast for both read and write operations and the
  resulting files are rather compact. Types are respected but all Python
  structures (like sets, lists and dictionaries) may not be directly
  exportable. You may need extra dependencies to read and write from this
  format.

The `Flight <traffic.core.flight.html>`_ and `Traffic <traffic.core.traffic.html>`_ structures implement the following methods:

.. autoclass:: traffic.core.mixins.DataFrameMixin()
  :members: from_file, to_csv, to_json, to_pickle, to_hdf, to_parquet
