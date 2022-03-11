How to access flight plan information?
======================================

A :class:`~traffic.core.flightplan.FlightPlan` is a structure designed to parse
flight plans in the ICAO format. Access to such field may be complicated for non
operational fellows, who may have access to different sources of data.

EUROCONTROL DDR SO6 files
-------------------------

.. automodule:: traffic.data.eurocontrol.ddr.so6
    :members: SO6, SO6Flight
    :no-undoc-members:
    :show-inheritance:

EUROCONTROL B2B web services
----------------------------

The `NM B2B web services
<https://www.eurocontrol.int/service/network-manager-business-business-b2b-web-services>`_
is an interface provided by the EUROCONTROL Network Manager (NM) for
system-to-system access to its services and data, allowing users to
retrieve and use the information in their own systems.

.. warning::

    You have to own a B2B certificate granted by EUROCONTROL to get access to
    this data.

.. danger::

    Because of a hard compromise to make between documentation and
    confidentiality of data, any information which cannot be found or inferred
    from different public sources is obfuscated in the following examples.

.. jupyter-execute::
    :hide-code:

    def repr_html(self):
        raise AttributeError

    from traffic.data.eurocontrol.b2b.flight import FlightList
    from traffic.data.eurocontrol.b2b.flight import FlightPlanList
    from traffic.data.eurocontrol.b2b.flow import RegulationList

    FlightList._repr_html_ = property(repr_html)
    FlightList.max_rows = 2
    FlightList._obfuscate = ['flightId', 'EOBT']

    FlightPlanList._repr_html_ = property(repr_html)
    FlightPlanList.max_rows = 2
    FlightPlanList._obfuscate = ['flightId', 'EOBT', 'status']

    RegulationList._repr_html_ = property(repr_html)
    RegulationList.max_rows = 5
    del RegulationList.columns_options['start']
    del RegulationList.columns_options['stop']
    del RegulationList.columns_options['tvId']
    del RegulationList.columns_options['airspace']
    del RegulationList.columns_options['aerodrome']
    RegulationList.columns_options['...'] = dict()
    RegulationList._obfuscate = ['regulationId']

The main instance of this class is provided as:

.. jupyter-execute::

    from traffic.data import nm_b2b

A path to your certificate and your password must be set in the configuration file.

.. code:: python

    >>> import traffic
    >>> traffic.config_file
    PosixPath('/home/xo/.config/traffic/traffic.conf')

Then edit the following lines accordingly:

::

    [nmb2b]

    ## This section contains information to log in to Eurocontrol B2B services
    ## https://traffic-viz.github.io/data_sources/flightplans.html

    pkcs12_filename =  # full path to the certificate file
    pkcs12_password = 
    # mode = OPS  # default: PREOPS
    # version = 25.0.0  # default: 25.0.0

We provide a basic API for some NM web services. They have been implemented
on a per need basis: not everything is provided, but functionalities may be
added in the future:

- :meth:`~traffic.data.eurocontrol.b2b.NMB2B.aixm_dataset`
- :meth:`~traffic.data.eurocontrol.b2b.NMB2B.flight_list`
- :meth:`~traffic.data.eurocontrol.b2b.NMB2B.flight_search`
- :meth:`~traffic.data.eurocontrol.b2b.NMB2B.flight_get`
- :meth:`~traffic.data.eurocontrol.b2b.NMB2B.regulation_list`

.. tip::

    File a `feature request
    <https://github.com/xoolive/traffic/issues/new?assignees=&labels=enhancement&template=feature-request.md>`_
    if you wish to have access to more services provided by B2B.

.. automethod:: traffic.data.eurocontrol.b2b.NMB2B.aixm_dataset
.. automethod:: traffic.data.eurocontrol.b2b.NMB2B.flight_list
.. automethod:: traffic.data.eurocontrol.b2b.NMB2B.flight_search
.. automethod:: traffic.data.eurocontrol.b2b.NMB2B.flight_get
.. automethod:: traffic.data.eurocontrol.b2b.NMB2B.regulation_list

.. autoclass:: traffic.data.eurocontrol.b2b.flight.FlightInfo
    :members:
    :no-undoc-members:

.. autoclass:: traffic.data.eurocontrol.b2b.flight.FlightList
    :members:
    :no-undoc-members:

.. autoclass:: traffic.data.eurocontrol.b2b.flight.FlightPlanList
    :members:
    :no-undoc-members:

.. autoclass:: traffic.data.eurocontrol.b2b.flow.RegulationInfo
    :members:
    :no-undoc-members:

.. autoclass:: traffic.data.eurocontrol.b2b.flow.RegulationList
    :members:
    :no-undoc-members:
