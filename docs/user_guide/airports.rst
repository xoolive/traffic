How to access airport information, including runway information and apron layout?
=================================================================================

.. autoclass:: traffic.data.basic.airports.Airports
    :members:
    :no-inherited-members:
    :no-undoc-members:

Jupyter notebook representation
-------------------------------

Airports offer special display outputs in Jupyter Notebook based on their
geographic footprint.

.. jupyter-execute::

    from traffic.data import airports

    airports['EHAM']

OpenStreetMap representation
----------------------------


Altair representation
---------------------

It is also possible to benefit from their Altair geographical representation.

.. jupyter-execute::

   import altair as alt
   
   from traffic.data import airports
   from traffic.data.samples import belevingsvlucht
   
   chart = (
       alt.layer(
           airports["EHAM"].geoencode(runways=True, labels=True),
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
       .configure_text(font="Ubuntu", fontSize=12, fontWeight=400)
       .configure_title(
           anchor="start", font="Ubuntu", fontSize=16, fontWeight="bold"
       )
       .configure_view(stroke=None)
   )
   
   chart

Matplotlib representation
-------------------------

Leaflet representation
----------------------


See also:

Airports of the world