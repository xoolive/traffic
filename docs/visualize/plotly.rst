Visualize trajectories with Plotly
==================================

Data visualisation is integrated with the `Plotly
<https://plotly.com/python/>`__ library as long as it is installed as an
optional dependency.

You may install `traffic` with the `plotly` option, the `full` option, or
install Plotly manually:

.. code:: bash

    # at install time
    pip install traffic[plotly]  # or full

    # with poetry
    poetry install traffic -E plotly  # or -E full

    # or simply manually
    pip install plotly
    conda install -c conda-forge plotly

Maps
~~~~

traffic provides the same interface as the :meth:`plotly.express` module on
the :class:`~traffic.core.Flight` and :class:`~traffic.core.Traffic` classes.
All kwargs arguments are passed directly to the corresponding method.

- with :meth:`plotly.express.line_mapbox`:

  .. jupyter-execute::

    from traffic.data.samples import belevingsvlucht

    belevingsvlucht.line_mapbox(color="callsign")

- with :meth:`plotly.express.scatter_mapbox`:

  .. jupyter-execute::

    from traffic.data.samples import belevingsvlucht

    fig = belevingsvlucht.scatter_mapbox(
        color="altitude", width=600, height=600, zoom=6
    )
    fig.update_layout(margin=dict(l=50, r=0, t=40, b=40))

- as animations (perform resampling in advance and limit yourself to few points):

  .. jupyter-execute::

    from traffic.data.samples import belevingsvlucht

    belevingsvlucht.resample("1 min").scatter_mapbox(
        color="vertical_rate",
        range_color=[-4000, 4000],
        animation_frame="timestamp",
        width=600,
        height=600,
        zoom=6,
    )

It is also possible to combine elements by constructing a
:class:`plotly.graph_objects.Scattermapbox` object:

.. jupyter-execute::

  from traffic.data import airports

  import plotly.graph_objects as go

  # fig = go.Figure()  # if necessary, we can initiate a Figure and fill it later

  fig = belevingsvlucht.resample("1 min").scatter_mapbox(
      color="vertical_rate",
      range_color=[-2000, 2000],
      animation_frame="timestamp",
      width=600,
      height=600,
      zoom=6,
  )

  fig.add_trace(
      belevingsvlucht.Scattermapbox(
          mode="lines",
          line=dict(color="#f58518", width=1),
      )
  )

  fig.update_layout(
      width=600,
      height=600,
      margin=dict(l=50, r=0, t=40, b=40),
      mapbox=dict(
          style="carto-positron",
          zoom=7,
          center=airports["EHLE"].latlon_dict,
      ),
  )

Similar functions are available and bound with :meth:`plotly.express.line_geo`
and :meth:`plotly.express.scatter_geo`:

.. jupyter-execute::

  from traffic.data import airports
  from traffic.data.samples import belevingsvlucht

  fig = belevingsvlucht.line_geo(
      scope="europe",
      projection="conic conformal",
      center=airports["EHLE"].latlon_dict,
  )
  fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
  fig.update_geos(resolution=50, fitbounds="locations")
  fig
