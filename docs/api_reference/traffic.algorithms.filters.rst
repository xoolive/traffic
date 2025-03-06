traffic.algorithms.filters
==========================

.. jupyter-execute::
    :hide-output:
    :hide-code:

    import altair as alt
    from traffic.core import Flight
    from traffic.data.samples import noisy_landing

    raw_data = noisy_landing.assign(type="raw data")
    default = noisy_landing.filter("default").assign(type="default")
    aggressive = noisy_landing.filter("aggressive").assign(type="aggressive")

    domain = ["raw data", "default", "aggressive", "Kalman filter", "Kalman smoother"]

    features = [
        "altitude (in ft)",
        "track angle (in deg)",
        "vertical_rate (in ft/min)",
        "groundspeed (in kts)",
    ]


    def chart(filtered: "Flight", titles: list[str] = features) -> alt.Chart:
        def y_channel(title: str):
            return alt.Y(title.split()[0], scale=alt.Scale(zero=False), title=None)

        color_channel = alt.Color(
            "type",
            scale=alt.Scale(domain=domain),
            legend=alt.Legend(title="Filter type"),
        )
        reference = raw_data.chart()
        compared = filtered.chart()
        return (
            alt.vconcat(
                alt.concat(
                    alt.layer(
                        reference.encode(y_channel(titles[0]), color_channel),
                        compared.encode(y_channel(titles[0]), color_channel),
                    ).properties(width=300, height=150, title=titles[0]),
                    alt.layer(
                        reference.encode(y_channel(titles[1]), color_channel),
                        compared.encode(y_channel(titles[1]), color_channel),
                    ).properties(width=300, height=150, title=titles[1]),
                ),
                alt.concat(
                    alt.layer(
                        reference.encode(y_channel(titles[2]), color_channel),
                        compared.encode(y_channel(titles[2]), color_channel),
                    ).properties(width=300, height=150, title=titles[2]),
                    alt.layer(
                        reference.encode(y_channel(titles[3]), color_channel),
                        compared.encode(y_channel(titles[3]), color_channel),
                    ).properties(width=300, height=150, title=titles[3]),
                ),
            )
            .configure_axisX(format="%H:%M", title=None, labelFontSize=12)
            .configure_axisY(labelFontSize=12)
            .configure_legend(orient="bottom", labelFontSize=14, titleFontSize=14)
            .configure_title(anchor="start", fontSize=16)
        )

traffic comes with some pre-implemented filters to be passed to the
:meth:`~traffic.core.Flight.filter` method. The method takes either a
:class:`~traffic.algorithms.filters.FilterBase` instance, or a string parameter:

- ``"default"`` is a relatively fast option with decent performance on
  trajectories extracted from the OpenSky database (with their most common
  glitches)

.. code:: python

    noisy_landing.filter()

.. jupyter-execute::
    :hide-code:

    chart(default)

- ``"aggressive"`` is a composition of several filters which may result in
  smoother trajectories.

.. code:: python

    noisy_landing.filter("aggressive")

.. jupyter-execute::
    :hide-code:

    chart(aggressive)

- :class:`~traffic.algorithms.filters.kalman.KalmanFilter6D` is a Kalman filter
  applied to the 6D state vector (latitude, longitude, altitude, track angle,
  groundspeed, vertical rate)

.. code:: python

    from traffic.algorithms.filters.kalman import KalmanFilter6D

    # The Kalman filter needs first a projection in x, y
    from cartes.crs import EuroPP

    noisy_landing.compute_xy(EuroPP()).filter(KalmanFilter6D())

.. jupyter-execute::
    :hide-code:

    from traffic.algorithms.filters.kalman import KalmanFilter6D

    noisy_landing_xy = noisy_landing.compute_xy()
    kalman = noisy_landing_xy.filter(KalmanFilter6D()).assign(type="Kalman filter")
    chart(kalman)

- :class:`~traffic.algorithms.filters.kalman.KalmanSmoother6D` is a Kalman
  smoother (a two-pass filter averaging the covariance of the errors on both
  sides) applied to the 6D state vector (latitude, longitude, altitude, track
  angle, groundspeed, vertical rate)

.. code:: python

    from traffic.algorithms.filters.kalman import KalmanSmoother6D

    noisy_landing.compute_xy(EuroPP()).filter(KalmanSmoother6D())

.. jupyter-execute::
    :hide-code:

    from traffic.algorithms.filters.kalman import KalmanSmoother6D

    smoother = noisy_landing_xy.filter(KalmanSmoother6D()).assign(type="Kalman smoother")
    chart(smoother)

API reference
-------------

.. automodule:: traffic.algorithms.filters
    :members:
    :show-inheritance:

.. automodule:: traffic.algorithms.filters.aggressive
    :members:
    :show-inheritance:

.. automodule:: traffic.algorithms.filters.consistency
    :members:
    :show-inheritance:

.. automodule:: traffic.algorithms.filters.kalman
    :members:
    :show-inheritance:
