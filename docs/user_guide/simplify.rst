How to simplify or resample a trajectory?
=========================================

ADS-B trajectories consist of a series of geographical coordinates and other
physical features relevant to the dynamics of the aircraft. Such *time series*
may be unnecessarily precise in some cases (e.g. straight lines or static
aircraft on ground), or contain gaps because of lack of coverage in terms of
data receivers.

The traffic library contains functions to deal with such issues, illustrated
here with the following :ref:`sample trajectory <How to access sample
trajectories?>`:

.. jupyter-execute::

    from traffic.data.samples import texas_longhorn

    texas_longhorn

.. contents:: Table of contents
   :depth: 1
   :local:

Trajectory simplification with Douglas-Peucker algorithm
--------------------------------------------------------

Trajectory simplification is particularly relevant for the ground track of
trajectories. The two most famous methods are the `Ramer-Douglas-Peucker
<https://bost.ocks.org/mike/simplify/>`__ algorithm and the `Visvalingam-Whyatt
<https://www.jasondavies.com/simplify/>`__ algorithm.

.. automethod:: traffic.core.flight.Flight.simplify

.. jupyter-execute::

    texas_longhorn.simplify(1e3) | texas_longhorn.simplify(1e4)

.. warning::

    A basic 2D simplification may result in an undesired oversimplification on
    another channel, esp. the altitude. An extra parameter ``altitude`` is
    available to help to control the simplification for that other feature. The
    ``z_factor`` is designed to give more or less weight to that feature with
    respect to the 2D ground track.

.. jupyter-execute::
    :code-below:

    import altair as alt

    encoding = [
        alt.X(
            "utcyearmonthdatehoursminutesseconds(timestamp):T",
            axis=alt.Axis(format="%H:%M", title=None),
        ),
        alt.Y("altitude", title="altitude (in ft)"),
        alt.Color("name:N", title="Simplification"),
    ]

    base = texas_longhorn.chart().transform_calculate(name="'original'")

    chart = (
        alt.vconcat(
            alt.layer(
                base.encode(*encoding),
                texas_longhorn.simplify(1e4)
                .chart()
                .transform_calculate(name="'simplify (2D)'")
                .encode(*encoding),
            ).properties(height=200, width=450),
            alt.layer(
                base.encode(*encoding),
                # specify a parameter to take the altitude into account
                texas_longhorn.simplify(1e4, altitude="altitude", z_factor=30.48)
                .chart()
                .transform_calculate(name="'simplify (3D)'")
                .encode(*encoding),
            ).properties(height=200, width=450),
        )
        .configure_axis(
            titleAngle=0,
            titleY=-15,
            titleX=0,
            titleAnchor="start",
            titleFont="Lato",
            titleFontSize=16,
            labelFontSize=12,
        )
        .configure_legend(
            orient="bottom",
            labelFont="Lato",
            labelFontSize=14,
            titleFont="Lato",
            titleFontSize=14,
        )
    )
    chart

Trajectory resampling with interpolation
----------------------------------------

Trajectory resampling is a useful tool in the following situations:

- reducing the number of samples and size of the resulting object;
- filling missing data in trajectory (interpolation);
- fitting many trajectories into vectors of equal length.


.. automethod:: traffic.core.flight.Flight.resample

.. jupyter-execute::

    texas_longhorn.resample("1 min") | texas_longhorn.resample("3 min")


Unlike with simplification, resampling is not designed to respect the topology
of a trajectory. However, trends in other features (like altitude) are better
preserved.

.. jupyter-execute::
    :code-below:

    encoding = [
        alt.X(
            "utcyearmonthdatehoursminutesseconds(timestamp):T",
            axis=alt.Axis(format="%H:%M", title=None),
        ),
        alt.Y("altitude", title="altitude (in ft)"),
        alt.Color("name:N", title="Resampling"),
    ]

    base = texas_longhorn.chart().transform_calculate(name="'original'")

    chart = (
        alt.vconcat(
            alt.layer(
                base.encode(*encoding),
                texas_longhorn.resample("1 min")
                .chart()
                .transform_calculate(name="'resample (1 min)'")
                .encode(*encoding),
            ).properties(height=200, width=450),
            alt.layer(
                base.encode(*encoding),
                texas_longhorn.resample("5 min")
                .chart()
                .transform_calculate(name="'resample (5 min)'")
                .encode(*encoding),
            ).properties(height=200, width=450),
        )
        .configure_axis(
            titleAngle=0,
            titleY=-15,
            titleX=0,
            titleAnchor="start",
            titleFont="Lato",
            titleFontSize=16,
            labelFontSize=12,
        )
        .configure_legend(
            orient="bottom",
            labelFont="Lato",
            labelFontSize=14,
            titleFont="Lato",
            titleFontSize=14,
        )
    )
    chart

.. tip::

    In many use cases, it is necessary to align trajectories in a specific frame
    (e.g. final approach) and get the same number of samples per trajectory.

In the following example, we resample many final approaches to Paris-Orly
airport with 30 samples per trajectory.  The x-axis displays the distance to
runway threshold; because of different ground speeds, all samples are not
perfectly aligned on the x-axis, but they are all equally distributed along the
time axis.

.. jupyter-execute::
    :code-below:

    from traffic.data.samples import quickstart

    landing_lfpo = (
        quickstart.next("aligned_on_ils('LFPO')")
        .cumulative_distance(reverse=True)
        .query("cumdist < 10")
        .resample(30)
        .eval()
    )

    chart = (
        alt.layer(
            *list(
                flight.chart()
                .mark_line(point=alt.OverlayMarkDef(color="#f58518"))
                .encode(
                    alt.X(
                        "cumdist",
                        scale=alt.Scale(reverse=True),
                        title="Cumulative distance to runway threshold (in nm)",
                    ),
                    alt.Y(
                        "altitude",
                        scale=alt.Scale(domain=(0, 4000)),
                        axis=alt.Axis(
                            title="Altitude (in ft)",
                            titleAngle=0,
                            titleY=-15,
                            titleX=0,
                            titleAnchor="start",
                        ),
                    ),
                )
                for flight in landing_lfpo
            )
        )
        .properties(height=250)
        .configure_axis(
            titleFont="Lato",
            titleFontSize=16,
            labelFontSize=12,
        )
    )

    chart
