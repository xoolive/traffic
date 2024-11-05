How to access flight plan information?
======================================

A :class:`~traffic.core.flightplan.FlightPlan` is a structure designed to parse flight plans in the ICAO format. Access to such field may be complicated for non-operational fellows, who may have access to different sources of data.

EUROCONTROL B2B web services
----------------------------

The `NM B2B web services <https://www.eurocontrol.int/service/network-manager-business-business-b2b-web-services>`_ is an interface provided by the EUROCONTROL Network Manager (NM) for system-to-system access to its services and data, allowing users to retrieve and use the information in their own systems.

.. danger::

    This code has been moved out of the traffic library. You may install `pyb2b <https://github.com/xoolive/pyb2b>` for the same functionalities.
