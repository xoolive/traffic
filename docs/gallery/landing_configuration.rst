Landing configurations
======================

A visualisation attempt at showing which landing configurations are the most
commonly used at Zurich airport. The analysis uses one of the datasets made
available in the traffic library.

.. jupyter-execute::
    :code-below:

    from traffic.data.datasets import landing_zurich_2019

    import altair as alt

    stats = (
        landing_zurich_2019.before("2019-10-10 00:00Z")
        .all("aligned_on_ils('LSZH')", flight_id="{self.flight_id}_{i}")
        .eval()
        .summary(["flight_id", "stop", "ILS_max"])
        .eval()
        .sort_values("stop")
    )

    chart = (
        alt.Chart(stats)
        .encode(
            alt.X("utchours(stop)", title=""),
            alt.Y("ILS_max", title=""),
            alt.Row("utcmonthdate(stop)", title=""),
            alt.Color("ILS_max", legend=None),
            alt.Size("count()", scale=alt.Scale(type="symlog"), title="Number of landings"),
        )
        .mark_circle()
        .properties(width=600, height=50)
        .configure_legend(orient="bottom")
        .configure_header(
            labelOrient="top", labelAnchor="start", labelFontWeight="bold", labelFontSize=12
        )
        .resolve_scale(x="independent", y="independent")
    )

    display(chart)
    display(stats)
