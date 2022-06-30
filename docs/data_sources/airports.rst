How to access airport information?
==================================

The traffic library provides an airport database with facilitated access to
**runway information** and **apron layout**.

.. autoclass:: traffic.data.basic.airports.Airports
    :members: search, extent
    :special-members: __getitem__
    :no-inherited-members:
    :no-undoc-members:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # hide the progress bars...
    from traffic.data import airports
    airports['EHAM']

Jupyter representation
----------------------

Airports offer special display outputs in Jupyter Notebook based on their
geographic footprint.

.. jupyter-execute::

    from traffic.data import airports
    airports['EHAM']

OpenStreetMap representation
----------------------------

The ``._openstreetmap()`` method queries information about the airport layout
from OpenStreetMap thanks to the `cartes <https://cartes-viz.github.io/>`_
library. This data is intended to be accessed through attributes corresponding
to ``aeroway`` tags, e.g. ``apron``, ``taxiway``, ``parking_position``,
``gate``, etc.

If you spot any error in data, consider contributing to OpenStreetMap.

.. jupyter-execute::

    airports["EHAM"].parking_position.head()

.. jupyter-execute::

    airports["EHAM"].taxiway.head()

.. warning::

    The GeoDataFrame associated with the represented data is accessible through
    the ``.data`` attribute.

Altair representation
---------------------

| It is also possible to benefit from their Altair geographical representation.
| Airports are associated with three parameters:

- ``footprint`` is True by default and shows the OpenStreetMap representation;
- | ``runways`` is True by default and represents the runways in bold lines.
  | Default parameters are ``strokeWidth=4, stroke="black"`` which can be
    overridden if a dictionary is passed in place of the boolean;
- | ``labels`` is True by default and represents the runway numbers.
  | Labels are automatically rotated along the runway bearing.
  | Default parameters are ``baseline="middle", dy=20, fontSize=18`` which can
    be overridden if a dictionary is passed in place of the boolean;

.. jupyter-execute::

   import altair as alt
   
   from traffic.data import airports
   from traffic.data.samples import belevingsvlucht
   
   chart = (
       alt.layer(
           airports["EHAM"].geoencode(
               footprint=True,  # True by default
               runways=True,  # default parameters
               labels=dict(fontSize=12, fontWeight=400, dy=10),  # font adjustments
           ),
           belevingsvlucht.first(minutes=1)
           .geoencode()
           .mark_line(color="steelblue"),
           belevingsvlucht.last(minutes=6)
           .geoencode()
           .mark_line(color="orangered"),
       )
       .properties(
           width=500, height=500, title="Belevingsvlucht at Amsterdam airport"
       )
       .configure_title(
           anchor="start", font="Lato", fontSize=16, fontWeight="bold"
       )
       .configure_view(stroke=None)
   )
   
   chart

Matplotlib representation
-------------------------

The airport representation can also be used in Matplotlib representations with a
similar approach.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from cartes.crs import Amersfoort  # Official projection in the Netherlands

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw=dict(projection=Amersfoort())
    )
    airports["EHAM"].plot(
        ax,
        footprint=True,
        runways=dict(color="#f58518"),  # update default parameters
        labels=dict(fontsize=12),
    )
    ax.spines["geo"].set_visible(False)


It is possible to adjust all parameters with parameters passed to the
``footprint`` (one entry per ``aeroway`` tag type), ``runways`` and ``labels``
arguments.

.. jupyter-execute::

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw=dict(projection=Amersfoort())
    )
    airports["EHAM"].plot(
        ax,
        footprint=dict(taxiway=dict(color="#f58518")),
        labels=dict(fontsize=12, color="crimson"),
    )
    ax.spines["geo"].set_visible(False)
