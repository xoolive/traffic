Trajectory generation
=====================


Introduction
------------

This library provide a Generation class for creating synthetic traffic data.
It implements fit() and sample() methods that call the corresponding methods
in the generative model passed as argument (``generation``).

You can import this class with the following code:

.. jupyter-execute::

    from traffic.algorithms.generation import Generation

.. jupyter-execute::
    :hide-code:

    import numpy as np

    np.random.seed(42)

Training models
---------------

In the case the generative model within your Generation object is not fitted
to any Traffic, you can use the fit() method.
Depending on the generative model used the fit() method can take some time, 
specifically if you use a deep generative model.

We load here a traffic data of landing trajectories at Zurich airport coming
from the north.

.. jupyter-execute::
    :hide-code:

    from traffic.core import Flight

    def coming_from_north(flight: Flight) -> bool:
        return (
            flight.data.iloc[0].track > 126 and 
            flight.data.iloc[0].track < 229 and
            flight.min("longitude") > 8.20 and
            flight.max("longitude") < 8.89 and
            flight.min("latitude") > 47.47 and
            flight.max("latitude") < 48.13
        )

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from traffic.data.datasets import landing_zurich_2019
    from traffic.core.projection import EuroPP

    t = (
        landing_zurich_2019
        .query("runway=='14'")
        .assign_id()
        .filter_if(coming_from_north)
        .resample(100)
        .unwrap()
        .eval(max_workers=1)
    )

    with plt.style.context("traffic"):
        ax = plt.axes(projection=EuroPP())
        t.plot(ax, alpha=0.05)
        t.centroid(nb_samples=None, projection=EuroPP()).plot(
            ax, color="red",alpha=1
        )

Before any fitting we enrich the Traffic DataFrame with the features we might
want to use to generate trajectories. 

For example, instead of working with ``longitude`` and ``latitude`` values,
we can compute their projection (``x`` and ``y`` respectively).

.. jupyter-execute::

    t = t.compute_xy(projection=EuroPP())

To keep track of time we propose to compute a ``timedelta`` parameter which is
for each trajectory coordinates, the difference in seconds with the beginning
of the trajectory.

.. jupyter-execute::

    from traffic.core import Traffic

    t = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in t
    )

Now we can use the fit() method to fit our generative model, here a Gaussian
Mixture with two components.

.. jupyter-execute::

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MinMaxScaler

    g1 = Generation(
        generation=GaussianMixture(n_components=1),
        features=["x", "y", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1))
    ).fit(t)

You can also use an API in the Traffic class to fit your model:

.. jupyter-execute::

    g2 = t.generation(
        generation=GaussianMixture(n_components=1),
        features=["x", "y", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1))
    )

.. warning::
    Make sure the generative model you want to use implements fit() and
    sample() methods.

.. note::
    The following codes are equivalent: ``t.generation(...)`` and
    ``Generation(...).fit(t)``.

Then we can sample the fitted model to produce new Traffic data.

.. jupyter-execute::

    t_gen1 = Traffic(
        g1.sample(
            500,
            projection=EuroPP(),
        )
    )
    t_gen2 = Traffic(
        g2.sample(
            500,
            projection=EuroPP(),
        )
    )

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=EuroPP()))
        t_gen1.plot(ax[0], alpha=0.1)
        t_gen1.centroid(nb_samples=None, projection=EuroPP()).plot(
            ax[0], color="red",alpha=1
        )
        t_gen2.plot(ax[1], alpha=0.1)
        t_gen2.centroid(nb_samples=None, projection=EuroPP()).plot(
            ax[1], color="red",alpha=1
        )

Do not forget to save the model if you want to use it later.

.. jupyter-execute::

    g1.save("_static/saved_model.pkl")

Loading models
--------------

It is possible to load a Generation object from a pickle file using the
from_file() method.

.. jupyter-execute::

    g = Generation.from_file("_static/saved_model.pkl")
    print(g)

Then you can either use the model to sample new trajectories or fit it on
another traffic.

Metrics
-------
