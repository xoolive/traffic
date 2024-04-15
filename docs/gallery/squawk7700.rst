Analysing in-flight emergencies using big data
----------------------------------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3937483.svg
    :target: https://doi.org/10.5281/zenodo.3937483

.. admonition:: Corresponding publication

    | X. Olive, A. Tanner, M. Strohmeier, M. Schäfer, M. Feridun, A. Tart, I. Martinovic and V. Lenders.
    | **OpenSky Report 2020: Analysing in-flight emergencies using big data**.
      `(paper) <http://www.cs.ox.ac.uk/files/12039/OpenSky%20Report%202020.pdf>`__
    | *Proceedings of the 39th Digital Avionics Systems Conference* (DASC), 2020


The dataset presented in the paper is available as a direct import in the library.

.. jupyter-execute::

    from traffic.data.datasets import squawk7700

    squawk7700

Metadata
========

Associated metadata is merged into the Traffic structure, but also
available as an attribute. The table includes information about:

-  flight information: ``callsign``, ``number`` (IATA flight number),
   ``origin`` , ``destination`` (where the aircraft intended to land),
   ``landing`` (where the aircraft actually landed, if available),
   ``diverted`` (where the aircraft actually landed, if applicable);
-  aircraft information: ``icao24`` transponder identifier,
   ``registration`` (the tail number) and ``typecode``;
-  information about the nature of the emergency, from Twitter and `The
   Aviation Herald <https://avherald.com/>`__.

.. jupyter-execute::

    squawk7700.metadata.iloc[:10, :10]  # just a preview to fit this page

.. jupyter-execute::

    squawk7700.metadata.iloc[:10, 10:]  # just a preview to fit this page


Data exploration
================

Simple queries provide subsets of the trajectories:

-  diverted aircraft: ``diverted == diverted`` selects flights where
   ``diverted`` is not empty (``NaN``);
-  returning aircraft, when the diversion airport is the origin airport

.. jupyter-execute::

    squawk7700.query("diverted == diverted") | squawk7700.query("diverted == origin")

For example, we can pick the following emergency situation:

.. jupyter-execute::

    squawk7700["AFR1196_20180303"]


The ``highlight`` keyword helps identifying parts of the trajectory
where the 7700 squawk code was activated.

.. jupyter-execute::

    from ipywidgets import Layout

    squawk7700["AFR1196_20180303"].map_leaflet(
        zoom=7,
        highlight=dict(red=lambda f: f.emergency()),
        layout=Layout(height="500px", max_width="800px"),
    )

Information about the nature of the emergencies have been collected from
two sources of information: Twitter and `The Aviation Herald <https://avherald.com>`_. The
following categories have been created:

-  ``nan`` means no information was found;
-  ``unclear`` means that we found an entry about the flight, but that
   the reason remains unknown;
-  ``misc`` means that the explanation does not fit any category.

.. jupyter-execute::

    tweet_issues = set(squawk7700.metadata.tweet_problem)
    avh_issues = set(squawk7700.metadata.avh_problem)
    tweet_issues | avh_issues

Cabin depressurisation
======================

Since the metadata has been merged into the Traffic structure, we can
select flights meeting certain requirements:

-  we found 31 flights related to cabin pressure or cracked windshields;
-  among them, 27 flights were diverted

.. jupyter-execute::

    pressure_pbs = ["cabin_pressure", "cracked_windshield"]
    pressure = squawk7700.query(
        f"tweet_problem in {pressure_pbs} or avh_problem in {pressure_pbs}"
    )
    pressure | pressure.query("diverted == diverted")


These flights are usually characterised by a rapid descent to around
10,000ft.

.. jupyter-execute::

    squawk7700["RPA4599_20190719"]

.. jupyter-execute::

    import altair as alt

    base = (
        squawk7700["RPA4599_20190719"]
        .chart()
        .encode(
            alt.X(
                "utchoursminutesseconds(timestamp)",
                axis=alt.Axis(title=None, format="%H:%M"),
            ),
            alt.Y(
                "altitude",
                title="altitude (in ft)",
                axis=alt.Axis(titleAngle=0, titleY=-15, titleAnchor="start"),
            ),
        )
    )

    chart = (
        alt.layer(
            base,
            base.transform_filter("datum.squawk == 7700").mark_line(color="#f58518"),
        )
        .properties(height=250)
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    chart

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # chart.save("_static/squawk7700-thumb.png")

Dumping fuel to reduce landing weight
=====================================

Emergencies are sometimes associated to dumping fuel in order to reduce
landing weight:

.. jupyter-execute::

    tweet_fueldump = set(squawk7700.metadata.tweet_fueldump)
    avh_fueldump = set(squawk7700.metadata.avh_fueldump)
    tweet_fueldump | avh_fueldump


.. jupyter-execute::

    fuel = ["fueldump", "hold_to_reduce"]
    squawk7700.query(f"tweet_fueldump in {fuel} or avh_fueldump in {fuel}")

.. jupyter-execute::

    squawk7700["AFL2175_20190723"] | squawk7700["BAW119_20190703"]


Landing attempts
================

Also, sometimes emergency situations are associated to several landing attempts,
at the same or at different airports.
:meth:`~traffic.core.Flight.aligned_on_ils` and
:meth:`~traffic.core.Flight.landing_attempts` are two methods available to
detect these events:

.. jupyter-execute::

    squawk7700["AFR1145_20190820"].last("45 min")

.. jupyter-execute::

    squawk7700["AFR1145_20190820"].landing_attempts()

.. jupyter-execute::

    squawk7700["AFR1145_20190820"].map_leaflet(
        zoom=9,
        airport="ELLX",
        highlight=dict(red=lambda f: f.landing_attempts()),
    )

| Explanation about this particular situation is available:
| `Incident: France A319 near Luxembourg on Aug 20th 2019, hot brakes indication
  <https://avherald.com/h?article=4cbcbfb7>`_

.. jupyter-execute::

    squawk7700.metadata.query('flight_id == "AFR1145_20190820"').iloc[0].to_dict()
