How to find flight phases on a trajectory?
==========================================

.. jupyter-execute::

    from traffic.data.samples import belevingsvlucht
    import altair as alt

    belevingsvlucht.phases().resample("10s").chart().encode(
        alt.X("utchoursminutes(timestamp):T").title(None),
        alt.Y("altitude").scale(domainMin=0),
        alt.Color("phase")
        .scale(
            domain=["LEVEL", "CLIMB", "DESCENT"],
            range=["#f58518", "#4c78a8", "#54a24b"],
        )
        .legend(orient="bottom"),
    ).mark_circle().properties(width=500)
