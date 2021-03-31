Landing configurations
======================

A visualisation attempt at showing which landing configurations are the most commonly used at Zurich airport. Can you spot the day when summer time CEST changes to CET?

.. raw:: html

    <div id="landing_configuration"></div>

    <script type="text/javascript">
      var spec = "../_static/landing_configuration.json";
      vegaEmbed('#landing_configuration', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>

.. code:: python

    from traffic.data.datasets import landing_zurich_2019
    import altair as alt

    stats = (
        landing_zurich_2019.all("aligned_on_LSZH")
        .eval(desc="", max_workers=8)
        .rename(columns=dict(flight_id="old_flight_id"))
        .summary(["old_flight_id_max", "stop", "ILS_max"])
        .query("stop.dt.month == 10")
        .rename(columns=dict(ILS_max="ILS", old_flight_id_max="flight_id"))
        .sort_values("stop")
    )

    data = (
        stats.assign(hour=lambda df: df.stop.dt.round("1H"))
        .groupby(["ILS", "hour"])
        .agg(dict(stop="count"))
        .rename(columns=dict(stop="count"))
        .reset_index()
    )

    selection = alt.selection_multi(fields=["ILS"], bind="legend")

    alt.Chart(data).encode(
        alt.X("utchours(hour):O", title="hour of day (UTC)",),
        alt.Y("utcmonthdate(hour):O", title=""),
        color="ILS",
        size="count",
        opacity=alt.condition(selection, alt.value(0.9), alt.value(0.2)),
    ).mark_circle().add_selection(selection)