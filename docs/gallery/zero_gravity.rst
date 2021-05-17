Zero-gravity flights
====================


.. raw:: html

    <div id="zero_gravity"></div>

    <script type="text/javascript">
      var spec = "../_static/zero_gravity.json";
      vegaEmbed('#zero_gravity', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>


.. code:: python

    from traffic.data.samples import zero_gravity

    import altair as alt


    base = (
        zero_gravity.phases(45)
        .skip("45T")
        .first("45T")
        .chart()
        .encode(
            alt.X(
                "utchoursminutesseconds(timestamp)",
                axis=alt.Axis(title=None, format="%H:%M"),
            )
        )
    )

    altitude_data = base.encode(
        y=alt.Y(
            "altitude",
            scale=alt.Scale(domain=(15000, 30000)),
            axis=alt.Axis(
                title="altitude (in ft)",
                labelColor="#5276A7",
                titleColor="#5276A7",
            ),
        ),
    ).mark_line(color="#5276A7")

    ias_data = base.encode(
        y=alt.Y(
            "IAS",
            scale=alt.Scale(domain=(100, 350)),
            axis=alt.Axis(
                title="indicated airspeed (in kts)",
                labelColor="#F18727",
                titleColor="#F18727",
            ),
        ),
    ).mark_line(color="#F18727")


    (
        alt.vconcat(
            alt.layer(altitude_data, ias_data)
            .resolve_scale(y="independent")
            .properties(width=500, height=200),
            base.encode(
                y=alt.Y("phase", title=None),
                color=alt.Color("phase", legend=None),
            )
            .mark_point()
            .properties(width=500, height=75),
        )
        .configure(font="Ubuntu")
        .configure_axis(labelFontSize=14, titleFontSize=16)
    )
