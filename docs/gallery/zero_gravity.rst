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


    data = alt.Chart(
        zero_gravity.phases(45)
        .skip("45T")
        .first("45T")
        .data[["timestamp", "altitude", "IAS", "phase"]]
    )

    altitude_data = data.encode(
        x=alt.X("timestamp", axis=None),
        y=alt.Y(
            "altitude",
            scale=alt.Scale(domain=(15000, 30000)),
            axis=alt.Axis(
                title="altitude (in ft)",
                labelFontSize=14,
                labelColor="#5276A7",
                titleFontSize=16,
                titleColor="#5276A7",
            ),
        ),
    ).mark_line(color="#5276A7")

    ias_data = data.encode(
        x=alt.X("timestamp", axis=None),
        y=alt.Y(
            "IAS",
            scale=alt.Scale(domain=(100, 350)),
            axis=alt.Axis(
                title="indicated airspeed (in kts)",
                labelFontSize=14,
                labelColor="#F18727",
                titleFontSize=16,
                titleColor="#F18727",
            ),
        ),
    ).mark_line(color="#F18727")


    (
        alt.layer(altitude_data, ias_data)
        .resolve_scale(y="independent")
        .properties(width=500, height=200)
        & data.encode(
            x=alt.X("timestamp", axis=alt.Axis(titleFontSize=16, labelFontSize=14)),
            y=alt.Y("phase", axis=alt.Axis(title="", labelFontSize=14),),
            color=alt.Color("phase", legend=None),
        )
        .mark_point()
        .properties(width=500, height=75)
    ).configure(font="Ubuntu")

