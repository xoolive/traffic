How to select go-arounds from a set of trajectories?
====================================================

Go-arounds are situations when, for reasons of safety, stability or after an
instruction from ATC, aircraft aligned on a runway interrupt their approach,
initiate an overshoot and *go around* for another attempt to land at the same
airport---possibly on a different runway. The change in thrust and go around in
trajectory may occur before or after the runway threshold.

The traffic library provides a function to detect go-arounds in trajectories:

.. automethod:: traffic.core.Flight.go_around
    :noindex:

The function returns a :class:`~traffic.core.FlightIterator` as there may be
several go-arounds (i.e. more than two attempts to land) at a given airport.

Let's illustrate how the library works with this dataset of trajectories landing
at Zurich airport over two months in 2019.

.. jupyter-execute::
    :hide-output:

    from traffic.data.datasets import landing_zurich_2019

    subset = landing_zurich_2019.between("2019-10-09", "2019-10-16")
    for flight in subset:
        for segment in flight.go_around("LSZH"):
            break
        else:  # https://stackoverflow.com/a/654002/
            continue
        break  # only reachable from nested break

Internally, in order to detect a go around, the library looks at two landing
attempts with :meth:`~traffic.core.Flight.aligned_on_ils`, and ensures there is
a :ref:`climbing phase <How to find flight phases on a trajectory?>`,
characterising the overshoot, between the two attempts.

.. jupyter-execute::
    :code-below:

    import altair as alt

    base = (
        segment.phases().chart()
        .encode(
            alt.X(
                "utchoursminutesseconds(timestamp)",
                axis=alt.Axis(format="%H:%M", title=None),
            )
        )
    )

    alt.vconcat(
        base.encode(alt.Y("phase", title="Flight phase"), alt.Color("phase")).mark_point(),
        base.encode(alt.Y("ILS", title="Runway ILS"), alt.Color("ILS", legend=None))
        .mark_point()
        .transform_filter("datum.ILS==34"),
        base.encode(alt.Y("altitude", title="altitude (in ft)")).properties(height=150),
    ).resolve_scale(color="independent").configure_legend(title=None).configure_axis(
        titleAngle=0,
        titleY=-15,
        titleX=0,
        titleAnchor="start",
        titleFont="Lato",
        titleFontSize=14,
        labelFontSize=12,
    )

.. jupyter-execute::

    flight.map_leaflet(airport="LSZH", zoom=9, highlight=dict(red="go_around('LSZH')"))


Among methods applicable on a :class:`~traffic.core.FlightIterator`, the
:meth:`~traffic.core.FlightIterator.has` method returns ``True`` if the iterator
is not empty:

.. jupyter-execute::

    flight.go_around("LSZH").has()

There is also a :meth:`~traffic.core.Flight.has` method available on
:class:`~traffic.core.Flight` objects: it accepts functions returning a
:class:`~traffic.core.FlightIterator` or strings representing a call to a
:class:`~traffic.core.Flight` method:

.. jupyter-execute::

    flight.has('go_around("LSZH")')

This helps to stack operations on a :class:`~traffic.core.lazy.LazyTraffic`. The
following visualization is an attempt to show whether go-arounds tend to occur
on particular days or times of a day (we could look for a correlation with
weather conditions) or whether they are just sporadic events due to external
factors. To be honest, nothing clear comes out of this one.

.. jupyter-execute::

    import altair as alt

    # the desc= argument in eval() creates a progress bar
    goarounds = subset.has('go_around("LSZH")').eval(max_workers=4)
    summary = goarounds.summary(['callsign', 'registration', 'stop']).eval()

    alt.Chart(summary).mark_square(size=100).encode(
        alt.X("utchours(stop):T", title="Hour of day"),
        alt.Y("utcday(stop):T", title="Day of month"),
        alt.Color("count()", title="Number of go-arounds"),
    ).properties(height=100).configure_legend(orient="bottom")

A few aircraft perform several go-arounds before landing. All attempts are not
necessarily on the same runway, as exemplified below:

.. jupyter-execute::

    for flight in goarounds:
        if flight.go_around().sum() > 1:
            display(flight)

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from cartes.crs import EuroPP

    from traffic.data import airports

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=EuroPP()))

        idx = 0
        for flight in goarounds:
            if flight.go_around().sum() > 1:
                airports["LSZH"].plot(ax[idx], footprint=False, runways=True)
                flight.plot(ax[idx], color="#bab0ac")
                for segment in flight.aligned_on_ils("LSZH"):
                    res, *_ = segment.plot(
                        ax[idx],
                        lw=1.5,
                        color="#4c78a8" if segment.ILS_max == "14" else "#f58518",
                    )
                    segment.at_ratio(0.5).plot(ax[idx], color=res.get_color())

                ax[idx].set_extent(segment, buffer=0.2)

                idx += 1

Here, we somehow broke the principle of separation between visualization and
trajectory processing.  It is actually possible to create a collection of
trajectories with more than one go around (more than 2 landing attempts):

- either with the :meth:`~traffic.core.Traffic.from_flights` class method;
- or by creating a custom function and stacking it with the
  :meth:`~traffic.core.Flight.pipe` operator

.. jupyter-execute::

    def many_goaround(flight: 'Flight') -> bool:
        return flight.go_around("LSZH").sum() > 1

    goarounds.iterate_lazy().pipe(many_goaround).eval()


In the following example, we try to look at possible contributing factors
leading to many go-arounds for one of the identified situations, which includes
a runway configuration change:

- bars behind aircraft represent the duration of the final approach (aligned
  with ILS);
- the colour of the trail represents the number of landing attempts;
- the runway configuration change suggests possible tail or cross wind
  conditions which are well-known contributing factors for go-arounds.

.. jupyter-execute::
    :code-below:

    data = (
        landing_zurich_2019.between("2019-10-15 10:10", "2019-10-15 10:50")
        .all("aligned_on_LSZH", flight_id="{self.callsign}_{i}")
        .summary(["callsign", "ILS_max", "start", "stop"])
        .eval()
        .rename(columns=dict(start="final approach", stop="landing"))
    )

    base = alt.Chart(
        # add one column in the table to count the landing attempts
        data.merge(
            data.groupby("callsign")["landing"].count().rename("landing attempts"),
            left_on="callsign",
            right_index=True,
        )
    )

    chart = (
        (
            base.mark_rule(size=3, opacity=0.5).encode(
                alt.X(
                    "utchoursminutes(final approach)",
                    axis=alt.Axis(title=""),
                ),
                alt.X2("utchoursminutes(landing)"),
                alt.Y("landing:N", sort="-x", axis=None),
                alt.Color("landing attempts:N"),
            )
            + base.mark_text(baseline="middle", align="left", dx=12).encode(
                alt.X("utchoursminutes(landing)"),
                alt.Y("landing:N"),
                alt.Text("callsign"),
                alt.Color("landing attempts:N"),
            )
            + base.mark_text(baseline="middle", align="left", size=25, dy=1, dx=-8).encode(
                alt.X("utchoursminutes(landing)"),
                alt.Y("landing:N"),
                alt.Color("landing attempts:N", title="Number of landing attempts"),
                text=alt.value("✈"),
            )
        )
        .properties(width=600, height=150)
        .facet(row="ILS_max")
        .configure_axis(labelFontSize=14)
        .configure_header(
            labelFontSize=24,
            labelFont="Ubuntu",
            labelOrient="right",
            labelAngle=90,
            labelPadding=-100,
            title=None,
        )
        .configure_legend(orient="bottom", labelFontSize=13, titleFontSize=13)
        .configure_text(font="Ubuntu")
        .resolve_axis(y="independent")
    )

    chart
