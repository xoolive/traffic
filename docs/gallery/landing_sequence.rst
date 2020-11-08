Landing sequence
================

| One chart per runway.
| One line per landing attempt, starting when the aircraft aligns with the ILS.
| The color matches the number of go arounds.

See how the configuration changed after two aircraft failed their landing.

.. raw:: html

    <div id="landing_sequence"></div>

    <script type="text/javascript">
      var spec = "../_static/landing_sequence.json";
      vegaEmbed('#landing_sequence', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="0" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>callsign</th>
          <th>ILS</th>
          <th>final approach</th>
          <th>landing</th>
          <th>go around</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>AFL2390</td>
          <td>14</td>
          <td>2019-10-15 10:11:17+00:00</td>
          <td>2019-10-15 10:14:07+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>SWR53E</td>
          <td>14</td>
          <td>2019-10-15 10:13:10+00:00</td>
          <td>2019-10-15 10:15:46+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>SWR287A</td>
          <td>14</td>
          <td>2019-10-15 10:14:46+00:00</td>
          <td>2019-10-15 10:17:40+00:00</td>
          <td>2</td>
        </tr>
        <tr>
          <td>HBVTB</td>
          <td>14</td>
          <td>2019-10-15 10:21:21+00:00</td>
          <td>2019-10-15 10:24:01+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>BAW712T</td>
          <td>14</td>
          <td>2019-10-15 10:22:51+00:00</td>
          <td>2019-10-15 10:26:06+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>SWR135G</td>
          <td>14</td>
          <td>2019-10-15 10:24:14+00:00</td>
          <td>2019-10-15 10:27:31+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>EDW27</td>
          <td>14</td>
          <td>2019-10-15 10:25:56+00:00</td>
          <td>2019-10-15 10:29:06+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>SWR287A</td>
          <td>14</td>
          <td>2019-10-15 10:28:41+00:00</td>
          <td>2019-10-15 10:31:23+00:00</td>
          <td>2</td>
        </tr>
        <tr>
          <td>SWR46J</td>
          <td>14</td>
          <td>2019-10-15 10:30:06+00:00</td>
          <td>2019-10-15 10:33:21+00:00</td>
          <td>1</td>
        </tr>
        <tr>
          <td>NJE715R</td>
          <td>14</td>
          <td>2019-10-15 10:31:58+00:00</td>
          <td>2019-10-15 10:35:04+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>GSW4943</td>
          <td>14</td>
          <td>2019-10-15 10:34:04+00:00</td>
          <td>2019-10-15 10:36:47+00:00</td>
          <td>0</td>
        </tr>
        <tr>
          <td>SWR287A</td>
          <td>28</td>
          <td>2019-10-15 10:41:58+00:00</td>
          <td>2019-10-15 10:45:31+00:00</td>
          <td>2</td>
        </tr>
        <tr>
          <td>SWR46J</td>
          <td>28</td>
          <td>2019-10-15 10:43:29+00:00</td>
          <td>2019-10-15 10:48:12+00:00</td>
          <td>1</td>
        </tr>
        <tr>
          <td>TYW212F</td>
          <td>28</td>
          <td>2019-10-15 10:47:02+00:00</td>
          <td>2019-10-15 10:49:59+00:00</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>

.. code:: python

    from traffic.data.datasets import landing_zurich_2019
    import altair as alt

    data = (
        landing_zurich_2019.between("2019-10-15 10:10", "2019-10-15 10:50")
        .all("aligned_on_LSZH")
        .assign(go=lambda df: df.index_.max())
        .eval(desc="")
        .drop(columns=["flight_id"])
        .summary(["callsign", "ILS_max", "start", "stop", 'go_max'])
        .sort_values("stop")
        .rename(columns=
            dict(
                ILS_max="ILS",
                start="final approach",
                stop="landing",
                go_max="go around"
            )
        )
    )

    chart = alt.Chart(data)

    (
        (
            c.mark_rule(size=3).encode(
                alt.X("utchoursminutes(final approach)", axis=alt.Axis(title="",),),
                alt.X2("utchoursminutes(landing)"),
                alt.Y("landing:N", sort="descending", axis=None),
                alt.Color("go around:N"),
            )
            + c.mark_text(baseline="middle", align="left", dx=12).encode(
                alt.X("utchoursminutes(landing)"),
                alt.Y("landing:N"),
                alt.Text("callsign"),
                alt.Color("go around:N"),
            )
            + c.mark_text(baseline="middle", align="left", size=20, dy=1, dx=-8).encode(
                alt.X("utchoursminutes(landing)"),
                alt.Y("landing:N"),
                alt.Color("go around:N"),
                text=alt.value("✈"),
            )
        )
        .properties(width=500, height=150)
        .facet(row="ILS")
        .configure_axis(labelFontSize=14,)
        .configure_header(
            labelFontSize=24,
            labelFont="Ubuntu",
            labelOrient="right",
            labelPadding=-30,
            title=None,
        )
        .configure_legend(orient="top")
        .configure_text(font="Ubuntu")
    )
