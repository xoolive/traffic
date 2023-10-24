How to implement trajectory generation?
=======================================

(contribution by Adrien Lafage `@alafage <https://github.com/alafage/>`_)

This library provides a ``Generation`` class for creating synthetic traffic data.
It implements ``fit()`` and ``sample()`` methods that call the corresponding
methods in the generative model passed as argument.

You can import this class with the following code:

.. jupyter-execute::

    from traffic.algorithms.generation import Generation

To instantiate such an object you can pass those arguments:

* ``generation``: Any object implementing ``fit()`` and ``sample()`` methods. It will define the generative model to use.
* ``features``: The list of the features to represent a trajectory.
* ``scaler``: A scaler that is optional to make sure each feature weights the same during the fitting part.

.. jupyter-execute::
    :hide-code:

    import numpy as np

    np.random.seed(42)

In the case the generative model within your ``Generation`` object is not fitted
to any ``Traffic`` object, you can use the ``fit()`` method.  Depending on the
generative model used, its ``fit()`` method can be rather time-consuming, esp.
with neural network-based generative models.

We load here traffic data of landing trajectories at Zurich airport coming
from the north.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from traffic.data.datasets import landing_zurich_2019
    from cartes.crs import EuroPP

    t = (
        landing_zurich_2019
        .query("runway == '14' and initial_flow == '162-216'")
        .assign_id()
        .unwrap()
        .resample(100)
        .eval()
    )

    with plt.style.context("traffic"):
        ax = plt.axes(projection=EuroPP())
        t.plot(ax, alpha=0.05)
        t.centroid(nb_samples=None, projection=EuroPP()).plot(
            ax, color="#f58518"
        )

Before any fitting, we enrich the Traffic DataFrame with the features we might
want to use to generate trajectories. For example, instead of working with
``longitude`` and ``latitude`` values, we can compute their projection (``x``
and ``y`` respectively).

.. jupyter-execute::

    t = t.compute_xy(projection=EuroPP())

To keep track of time we propose to compute a ``timedelta`` parameter which is
for each trajectory coordinates, the difference in seconds with the beginning
of the trajectory.

.. jupyter-execute::

    from traffic.core import Traffic

    def compute_timedelta(df: "pd.DataFrame"):
        return (df.timestamp - df.timestamp.min()).dt.total_seconds()

    t = t.iterate_lazy().assign(timedelta=compute_timedelta).eval()

Now we can use the ``fit()`` method to fit our generative model, here a Gaussian
Mixture with two components.

.. jupyter-execute::

    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MinMaxScaler

    g1 = Generation(
        generation=GaussianMixture(n_components=2),
        features=["x", "y", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1, 1))
    ).fit(t)

.. note::

    This code is equivalent to the following call on the ``Traffic`` object:

    .. jupyter-execute::

        g2 = t.generation(
            generation=GaussianMixture(n_components=1),
            features=["x", "y", "altitude", "timedelta"],
            scaler=MinMaxScaler(feature_range=(-1, 1))
        )

.. warning::

    Make sure the generative model you want to use implements the ``fit()`` and ``sample()`` methods.

Then we can sample the fitted model to produce new Traffic data.

.. jupyter-execute::

    t_gen1 = g1.sample(500, projection=EuroPP())
    t_gen2 = g2.sample(500, projection=EuroPP())

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=EuroPP()))

        t_gen1.plot(ax[0], alpha=0.2)
        t_gen1.centroid(nb_samples=None, projection=EuroPP()).plot(
            ax[0], color="#f58518"
        )

        t_gen2.plot(ax[1], alpha=0.2)
        t_gen2.centroid(nb_samples=None, projection=EuroPP()).plot(
            ax[1], color="#f58518"
        )


.. warning::

    This very naive model obviously does not produce very convincing results. More appropriate methods will be provided in a near future.

.. autoclass:: traffic.algorithms.generation.Generation
    :members:
    :inherited-members:
    :no-undoc-members:
    :show-inheritance:
