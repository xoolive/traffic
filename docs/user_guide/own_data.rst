How to bring your own data in the traffic library?
==================================================

``Flight`` and ``Traffic`` wrap a pandas DataFrame without validating,
renaming, or converting anything: the DataFrame you pass in must already use
the column names the library expects.

Required columns
-----------------

The minimum set of columns, as described in the ``Flight`` class
documentation, is:

- ``icao24``: the ICAO transponder ID of an aircraft;
- ``callsign``: an identifier associated with the registration of an
  aircraft, its mission, or a route;
- ``timestamp``: timezone aware timestamps are preferable;
- ``latitude``, ``longitude``: in degrees, WGS84 (EPSG:4326);
- ``altitude``: in feet. Data recorded in meters must be converted before
  use, e.g. ``altitude_ft = altitude_m / 0.3048``.

A ``flight_id`` column may be used in place of the (``icao24``, ``callsign``)
pair, which is useful when a dataset has no ``callsign`` column.

Loading a CSV file
-------------------

Consider a CSV file with columns ``icao24``, ``datetime``, ``lat``, ``lon``
and ``alt``. The columns must be renamed, and the timestamp column must be
converted to a timezone aware datetime:

.. code-block:: python

    import pandas as pd
    from traffic.core import Flight

    df = pd.read_csv("example_flight.csv").rename(
        columns={
            "datetime": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "alt": "altitude",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    flight = Flight(df)

Timezone naive timestamps may work with some methods, but the behaviour is
not guaranteed: converting to UTC with ``utc=True`` avoids this.
