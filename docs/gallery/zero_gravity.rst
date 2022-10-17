Zero-gravity flights
====================

`Zero-gravity flights <https://www.airzerog.com/>`_ are regularly operated on an
Airbus A310 in order to conduct many kind of scientific experiments in zero
gravity.

One of such trajectories is present in the :ref:`sample flights <How to access
sample trajectories?>` of the library: we can observe the characteristic profile
in altitude and indicated airspeed.

.. jupyter-execute::
    :code-below:

    import altair as alt

    from traffic.data.samples import zero_gravity

    focus = zero_gravity.phases(twindow=45).skip("45 min").first("15 min")

    base = focus.chart().encode(
        alt.X(
            "utchoursminutesseconds(timestamp)",
            axis=alt.Axis(title=None, format="%H:%M"),
        )
    )

    chart = (
        alt.vconcat(
            alt.layer(
                base.encode(
                    y=alt.Y(
                        "altitude",
                        scale=alt.Scale(domain=(15000, 30000)),
                        axis=alt.Axis(
                            title="altitude (in ft)",
                            labelColor="#4c78a8",
                            titleColor="#4c78a8",
                        ),
                    ),
                ).mark_line(color="#4c78a8"),
                base.encode(
                    y=alt.Y(
                        "IAS",
                        scale=alt.Scale(domain=(100, 350)),
                        axis=alt.Axis(
                            title="indicated airspeed (in kts)",
                            labelColor="#f58518",
                            titleColor="#f58518",
                            titleAnchor="end",
                        ),
                    ),
                ).mark_line(color="#f58518"),
            )
            .resolve_scale(y="independent")
            .properties(width=500, height=200),
            base.encode(
                y=alt.Y("phase", title="Flight phase"),
                color=alt.Color(
                    "phase",
                    legend=None,
                    scale=alt.Scale(
                        domain=["LEVEL", "CLIMB", "DESCENT"],
                        range=["#4c78a8", "#f58518", "#54a24b"],
                    ),
                ),
            )
            .mark_point()
            .properties(width=500, height=75),
        )
        .configure_axis(
            labelFontSize=14,
            titleFontSize=16,
            titleAngle=0,
            titleY=-10,
            titleAnchor="start",
        )
    )
    chart

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # chart.save("_static/zero_gravity-thumb.png")


.. jupyter-execute::
    :code-below:

    from ipyleaflet import Map

    m = Map(zoom=8, center=(49, -2))
    m.add(zero_gravity, color="#79706e", weight=2)

    for segment in focus.query('phase=="CLIMB"').split("30s"):
        m.add(segment, color="#f58518", weight=4)
    for segment in focus.query('phase=="DESCENT"').split("30s"):
        m.add(segment, color="#54a24b", weight=4)
    for segment in focus.query('phase=="LEVEL"').split("30s"):
        m.add(segment, color="#4c78a8", weight=4)

    m
