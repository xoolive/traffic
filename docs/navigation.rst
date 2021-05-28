Navigation events
=================

The traffic library implements many functions to detect navigation events
relevant from an operational point of view. These methods are all built on basic
blocks and available on `Flight <traffic.core.flight.html>`__ structures.

This page organises those methods by scope:

- `TMA events <#tma-events>`__ refer to the airborne part of a trajectory in the terminal maneuvering area around an airport. They extract information from parts of trajectory before landing or after take-off;

- `airborne events <#airborne-events>`__ are more general events during the cruise phase;

- `ground events <#ground-events>`__ refer to all what happens when on ground at a given airport.


TMA events
----------

.. automethod:: traffic.core.Flight.takeoff_airport
.. automethod:: traffic.core.Flight.takeoff_from

.. automethod:: traffic.core.Flight.landing_airport
.. automethod:: traffic.core.Flight.landing_at


- ``aligned_on_ils()``
- ``takeoff_from_runway()``

- ``runway_change()``
- ``go_around()``


Airborne events
---------------

- ``aligned_on_navpoint()``
- ``compute_navpoints()``
- ``emergency()``

Ground events
-------------

- ``aligned_on_runway()``
- ``on_parking_position()``
- ``pushback()``
- ``slow_taxi()``
- ``moving()``