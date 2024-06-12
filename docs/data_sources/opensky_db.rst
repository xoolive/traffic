How to access ADS-B data from OpenSky history database?
=======================================================

For more advanced request or a dig in further history, you may be
eligible for an direct access to the history database through their
`Impala <https://opensky-network.org/impala-guide>`__ shell or Trino database.

.. warning::

  OpenSky data are subject to particular `terms of use
  <https://opensky-network.org/about/terms-of-use>`_. In particular, if you plan
  to use data for commercial purposes, you should mention it when you ask for
  access

Provided functions are here to help:

- build appropriate and efficient requests without any SQL knowledge;
- split requests efficiently and store intermediary results in cache
  files;
- parse results with pandas and wrap results in appropriate data structures.

The first thing to do is to put your credentials in your configuration
file. Edit the following lines of your configuration file.


.. parsed-literal::

    [opensky]
    username =
    password =

You can check the path to your configuration file here. The path is
different according to OS versions so do not assume anything and check
the contents of the variable.

.. code:: python

    >>> import traffic
    >>> traffic.config_file
    PosixPath('/home/xo/.config/traffic/traffic.conf')


Historical traffic data
-----------------------

.. automethod:: traffic.data.adsb.opensky.OpenSky.history

Examples of requests
~~~~~~~~~~~~~~~~~~~~

First, the `opensky` instance parses your configuration file upon import:

.. jupyter-execute::

    from traffic.data import opensky

Then you may send requests:

- based on callsign:

    .. jupyter-execute::

        flight = opensky.history(
            "2017-02-05 15:45",
            stop="2017-02-05 16:45",
            callsign="EZY158T",
            # returns a Flight instead of a Traffic
            return_flight=True
        )
        flight

- based on bounding box:

    .. code:: python

        # two hours of traffic over LFBB FIR
        t_lfbb = opensky.history(
            "2018-10-01 11:00",
            "2018-10-01 13:00",
            bounds=eurofirs['LFBB']
        )

- based on airports and callsigns (with wildcard):

    .. code:: python

        # Airbus test flights from and to Toulouse airport
        t_aib = opensky.history(
            "2019-11-01 09:00",
            "2019-11-01 12:00",
            departure_airport="LFBO",
            arrival_airport="LFBO",
            callsign="AIB%",
        )

- based on airport (with origin/destination ICAO id):

    .. code:: python

        # flights from and to Zurich airport
        t_lszh = opensky.history(
            start="2024-03-15 09:00",
            stop="2024-03-15 11:00",
            airport="LSZH",
            selected_columns=(
                # colums from StateVector4 (quoted or not)
                StateVectorsData4.time,
                'icao24', 'lat', 'lon', 'velocity', 'heading', 'vertrate',
                'callsign', 'onground', 'alert', 'spi', 'squawk', 'baroaltitude',
                'geoaltitude', 'lastposupdate', 'lastcontact', 'serials', 'hour',
                # (some) columns from FlightsData4: always quoted!
                # returned as columns 'estdepartureairport' and 'estarrivalairport'
                'FlightsData4.estdepartureairport', 'FlightsData4.estarrivalairport'),
        )


- based on (own?) receiver's identifier:

    .. code:: python

        t_sensor = opensky.history(
            "2019-11-11 10:00",
            "2019-11-11 12:00",
            serials=1433801924,
        )

Extended Mode-S (EHS)
---------------------

EHS messages are not automatically decoded for you in the OpenSky
Database but you may access them and decode them from your computer.

.. warning::

    **Some examples here may be outdated**. To our knowledge at this time, only
    EHS data **after January 1st 2020** are available!

.. tip::

    | ``Flight.query_ehs()`` messages also takes a dataframe argument to avoid
      making possibly numerous requests to the Impala database.
    | Consider using `opensky.extended()
      <#traffic.data.adsb.opensky.OpenSky.extended>`_ and request all
      necessary data, then pass the resulting dataframe as an argument.

.. automethod:: traffic.data.adsb.opensky.OpenSky.extended

Examples of requests
~~~~~~~~~~~~~~~~~~~~

- based on transponder identifier (icao24):

    .. code:: python

        from traffic.data.samples import belevingsvlucht

        df = opensky.extended(
            belevingsvlucht.start,
            belevingsvlucht.stop,
            icao24=belevingsvlucht.icao24
        )

        enriched = belevingsvlucht.query_ehs(df)

- based on geographical bounds:

    .. code:: python

        from traffic.data import eurofirs
        from traffic.data.samples import switzerland

        df = opensky.extended(
            switzerland.start_time,
            switzerland.end_time,
            bounds=eurofirs['LSAS']
        )

        enriched_ch = (
            switzerland
            .filter()
            .query_ehs(df)
            .resample('1s')
            .eval(desc='', max_workers=4)
        )

- based on airports, together with traffic:

    .. code:: python

        schiphol = opensky.history(
            "2019-11-11 12:00",
            "2019-11-11 14:00",
            airport="EHAM"
        )

        df = opensky.extended(
            "2019-11-11 12:00",
            "2019-11-11 14:00",
            airport="EHAM"
        )

        enriched_eham = (
            schiphol
            .filter()
            .query_ehs(df)
            .resample('1s')
            .eval(desc='', max_workers=4)
        )


Flight list by airport
----------------------

.. automethod:: traffic.data.adsb.opensky.OpenSky.flightlist

Requests for raw data
---------------------

.. automethod:: traffic.data.adsb.opensky.OpenSky.rawdata
