Web services from Eurocontrol NM
--------------------------------

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
