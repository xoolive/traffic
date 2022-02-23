How to access flight plan information?
======================================

:class:`~traffic.core.FlightPlan` are structures designed to parse flight plans
in the ICAO format. Access to such field may be complicated for non operational
fellows, who may have access to different sources of data.

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

We provide a basic API for some NM web services. They have been implemented
on a per need basis: not everything is provided, but functionalities may be
added in the future.

.. warning::

    You have to own a B2B certificate granted by EUROCONTROL to get access to
    this data.

.. automodule:: traffic.data.eurocontrol.b2b
    :members:
    :no-inherited-members:
    :no-undoc-members:

    .. automethod:: NMB2B.flight_search
    .. automethod:: NMB2B.flight_list
    .. automethod:: NMB2B.flight_get
    .. automethod:: NMB2B.regulation_list