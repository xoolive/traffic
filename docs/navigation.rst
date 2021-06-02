Navigation events
=================

The traffic library implements many functions to detect navigation events
relevant from an operational point of view. These methods are all built on basic
blocks and available on `Flight <traffic.core.flight.html>`__ structures.

This page organises those methods by scope:

- `TMA events <#tma-events>`__ refer to the airborne part of a trajectory in the terminal maneuvering area around an airport. They extract information from parts of trajectory before landing or after take-off;

- `airborne events <#airborne-events>`__ are more general events during the cruise phase;

- `ground events <#ground-events>`__ refer to all what happens when on ground at a given airport.


Basic metadata
--------------

.. automethod:: traffic.core.Flight.takeoff_airport
.. automethod:: traffic.core.Flight.takeoff_from
.. automethod:: traffic.core.Flight.landing_airport
.. automethod:: traffic.core.Flight.landing_at

TMA events
----------

.. automethod:: traffic.core.Flight.takeoff_from_runway
.. automethod:: traffic.core.Flight.aligned_on_ils
.. automethod:: traffic.core.Flight.go_around
.. automethod:: traffic.core.Flight.runway_change

Airborne events
---------------

.. automethod:: traffic.core.Flight.aligned_on_navpoint
.. automethod:: traffic.core.Flight.compute_navpoints
.. automethod:: traffic.core.Flight.emergency


Ground events
-------------

.. automethod:: traffic.core.Flight.aligned_on_runway
.. automethod:: traffic.core.Flight.on_parking_position
.. automethod:: traffic.core.Flight.pushback
.. automethod:: traffic.core.Flight.slow_taxi
.. automethod:: traffic.core.Flight.moving
