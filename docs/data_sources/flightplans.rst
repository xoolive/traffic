How to access flight plan information?
======================================

.. warning::

   Documentation under refactoring, still work in progress


Flight Plan structures
----------------------

.. autoclass:: traffic.core.FlightPlan
    :members:
    :no-undoc-members:
    :show-inheritance:

EUROCONTROL DDR SO6 files
-------------------------

.. autoclass:: traffic.data.SO6
    :members:
    :no-undoc-members:
    :show-inheritance:

.. autoclass:: traffic.data.eurocontrol.ddr.so6.Flight
    :members:
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

.. autoclass:: traffic.data.eurocontrol.b2b.NMB2B()
    :members:
    :no-inherited-members:
    :no-undoc-members:

    .. automethod:: flight_search
    .. automethod:: flight_list
    .. automethod:: flight_get
    .. automethod:: regulation_list