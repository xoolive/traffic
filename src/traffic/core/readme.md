# The `core` module

The `core` module consists of the basic hierarchy of classes and mixins.

## Mixins

Most traffic data classes are based on the `DataFrameMixin` mixin which embeds a `pandas.DataFrame` and provides basic representation method. Similarly the `ShapelyMixin` embeds a basic representation methods for geographic structures (trajectories, sectors, etc.).

## Traffic and Flight

A `Flight` object inherits from both mixins: it embeds a `pandas.DataFrame` with several columns to be expected: `latitude`, `longitude`, `altitude` and `timestamp` and provides specific methods for:

- accessing metadata of a given Flight, start date, end date, aircraft registration, etc.
- common GIS operations on trajectories: intersection with a sector, clipping within a sector, etc.
- exporting to various formats: csv, hdf5, matplotlib, kml, czml

A `Traffic` object is a structure flattening several flights. It provides methods for accessing every flight by identification (callsign, aircraft id or flight id) and unfolds Flight methods on all flights included.

## Sector

A `Sector` object is a list of `ShapelyMixin` objects associated with a lower and a upper altitude. A sector provides intersection and clipping methods to `Flight` objects.
