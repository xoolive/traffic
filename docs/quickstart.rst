Getting started
===============

.. jupyter-execute::
    :hide-code:
    :hide-output:

    %matplotlib inline

The motivation for this page/notebook is to take the reader through all
basic functionalities of the traffic library. In particular, we will cover:

1. a :ref:`basic introduction <basic introduction>` about
   :class:`~traffic.core.Flight` and :class:`~traffic.core.Traffic` structures;
2. how to produce :ref:`visualisations <data visualization>` of trajectory data;
3. a :ref:`use case <Low-altitude trajectory patterns in Paris metropolitan
   area>` to display trajectories around Paris area;
4. an introduction to :ref:`declarative trajectory processing <declarative
   trajectory processing>` through lazy iteration

.. tip::

    This page is also available as a notebook which can be
    :jupyter-download-notebook:`downloaded <quickstart>` and executed locally;
    or loaded and executed in `Google Colab
    <https://colab.research.google.com/>`__.

Basic introduction
------------------

The traffic library provides natural methods and attributes that can be
applied on trajectories and collection of trajectories, all represented
as pandas DataFrames.

Flight objects
~~~~~~~~~~~~~~

    :class:`~traffic.core.Flight` is the core class offering representations,
    methods and attributes to single trajectories.  Trajectories can either:

    - be imported from the :ref:`sample trajectory set <How to access sample
      trajectories?>`;
    - be downloaded from :ref:`OpenSky Impala shell <How to access ADS-B data
      from OpenSky history database?>`;
    - be loaded from a :ref:`tabular file <How to export and store trajectory
      and airspace data?>` (csv, json, parquet, etc.);
    - be decoded from :ref:`raw ADS-B signals <How to decode ADS-B and Mode S
      data?>`.

    The :ref:`belevingsvlucht` from the :ref:`sample trajectory set <How to
    access sample trajectories?>` is present throughout the documentation:

    .. jupyter-execute::

        from traffic.data.samples import belevingsvlucht

    Many representations are available:

    - in a Python interpreter:

        .. jupyter-execute::

            print(belevingsvlucht)

    - with `rich <https://rich.readthedocs.io/en/latest/>`_ simple or advanced
      representations:

        .. jupyter-execute::

            from rich.pretty import pprint
            pprint(belevingsvlucht)

        .. jupyter-execute::

            # the console is not necessary if you ran pretty.install()
            from rich.console import Console
            console = Console()
            console.print(belevingsvlucht)

    - in a Jupyter notebook:
        .. jupyter-execute::

            belevingsvlucht

    Information about each :class:`~traffic.core.Flight` is available through
    attributes or properties:

    .. jupyter-execute::

        dict(belevingsvlucht)


    Methods are provided to select relevant parts of the flight, e.g. based on
    timestamps. The :attr:`~traffic.core.Flight.start` and
    :attr:`~traffic.core.Flight.stop` properties refer to the timestamps of the
    first and last recorded samples. Note that all timestamps are by default set
    to universal time (UTC) as it is common practice in aviation.

    .. jupyter-execute::

        (belevingsvlucht.start, belevingsvlucht.stop)


    .. jupyter-execute::

        belevingsvlucht.first(minutes=30)

    .. warning::

        Note the difference between the "strict" comparison (:math:`>`) vs. "or
        equal" comparison (:math:`\geq`)

    .. jupyter-execute::

        belevingsvlucht.after("2018-05-30 19:00", strict=False)

    .. note::

        Each :class:`~traffic.core.Flight` is wrapped around a
        :class:`pandas.DataFrame`: when no method is available for your
        particular need, you can always access the underlying dataframe.

    .. jupyter-execute::

        belevingsvlucht.between("2018-05-30 19:00", "2018-05-30 20:00").data

Traffic objects
~~~~~~~~~~~~~~~

    :class:`~traffic.core.Traffic` is the core class to represent collections of
    trajectories.  In practice, all trajectories are flattened in the same
    :class:`pandas.DataFrame`.

    .. jupyter-execute::

        from traffic.data.samples import quickstart

    The basic representation of a :class:`~traffic.core.Traffic` object is a
    summary view of the data: the structure tries to infer how to separate
    trajectories in the data structure based on customizable heuristics, and
    returns a number of sample points for each trajectory.

    .. jupyter-execute::

        quickstart

    | :class:`~traffic.core.Traffic` objects offer the ability to **index** and
      **iterate** on all flights contained in the structure.
    | In order to separate and identify trajectories (:class:`~traffic.core.Flight`),
      :class:`~traffic.core.Traffic` objects will use either:

      -  a customizable flight identifier (``flight_id``); or
      -  a combination of ``timestamp`` and ``icao24`` (aircraft identifier);

    Indexation will be made on:

    - ``icao24``, ``callsign`` (or ``flight_id`` if available):

        .. jupyter-execute::

            quickstart["TAR722"]  # return type: Flight, based on callsign
            quickstart["39b002"]  # return type: Flight, based on icao24

    - an integer or a slice, to take flights in order in the collection:

        .. jupyter-execute::

            quickstart[0]  # return type: Flight, the first trajectory in the collection
            quickstart[:10]  # return type: Traffic, the 10 first trajectories in the collection


    - a subset of trajectories can also be selected:

        - if a list is passed an index:

          .. jupyter-execute::

            quickstart[['AFR83HQ', 'AFR83PX', 'AFR84UW', 'AFR91QD']]

        - with a pandas-like :meth:`~traffic.core.Traffic.query`:

          .. jupyter-execute::

            quickstart.query('callsign.str.startswith("AFR")')

    There are several ways to assign a flight identifier. The most simple one
    that you will use in 99% of situations involves the
    :meth:`~traffic.core.Flight.flight_id` method.

    .. jupyter-execute::

        quickstart.assign_id().eval()

    We will explain :ref:`further <Declarative trajectory processing>` what the
    :meth:`~traffic.core.lazy.LazyTraffic.eval()` method is about.

Data visualization
------------------

The traffic library offers facilities to leverage the power of common
visualization renderers including `Matplotlib <https://matplotlib.org/>`_ and
`Altair <https://altair-viz.github.io/>`__.

- with Matplotlib, the ``traffic`` style context (optional) offers a convenient
  initial stylesheet:

  .. jupyter-execute::

    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    with plt.style.context("traffic"):

        fig, ax = plt.subplots(figsize=(10, 7))

        (
            belevingsvlucht
            .between("2018-05-30 19:00", "2018-05-30 20:00")
            .plot_time(
                ax=ax,
                y=["altitude", "groundspeed"],
                secondary_y=["groundspeed"]
            )
        )

        ax.set_xlabel("")
        ax.tick_params(axis='x', labelrotation=0)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

- | The :meth:`~traffic.core.Flight.chart` method triggers an initial representation with Altair which can be further refined.
  | For example, with the following subset of trajectories:

  .. jupyter-execute::

    subset = quickstart[["TVF22LK", "EJU53MF", "TVF51HP", "TVF78YY", "VLG8030"]]

  .. jupyter-execute::
    :hide-code:
    :hide-output:

    subset = subset.query("altitude.isnull() or altitude < 20000")

  .. jupyter-execute::

    subset[0].chart()

  Even a simple visualization without a physical features plotted on the
  y-channel can be meaningful. The following proposition helps visualizing when
  aircraft are airborne:

  .. jupyter-execute::

      import altair as alt

      # necessary line if you see an error about a maximum number of rows
      alt.data_transformers.disable_max_rows()

      alt.layer(
          *(
              flight.chart().encode(
                  alt.Y("callsign", sort="x", title=None),
                  alt.Color("callsign", legend=None),
              )
              for flight in subset
          )
      ).configure_line(strokeWidth=4)

  The y-channel is however most often used to plot physical quantities such as
  altitude, ground speed, or more.

  .. jupyter-execute::

    alt.layer(
        *(
            flight.chart().encode(
                alt.Y("altitude"),
                alt.Color("callsign"),
            )
            for flight in subset
        )
    )

  Simple plots are beautiful by default, but it is still possible to further
  refine them. For more advanced tricks with Altair, refer to their `online
  documentation <https://altair-viz.github.io/>`_.

  .. jupyter-execute::

    chart = (
        alt.layer(
            *(
                flight.chart().encode(
                    alt.X(
                        "utcdayhoursminutesseconds(timestamp)",
                        axis=alt.Axis(format="%H:%M"),
                        title=None,
                    ),
                    alt.Y("altitude", title=None, scale=alt.Scale(domain=(0, 18000))),
                    alt.Color("callsign"),
                )
                for flight in subset
            )
        )
        .properties(title="altitude (in ft)")  # "trick" to display the y-axis title horizontally
        .configure_legend(orient="bottom")
        .configure_title(anchor="start", font="Lato", fontSize=16)
    )
    chart

Making maps
-----------

Maps are also available with Matplotlib, Altair, and thanks to `ipyleaflet
<https://ipyleaflet.readthedocs.io/>`_ widgets.

- with Matplotlib, you need to specify a projection for your axis system. They
  are provided by `cartes <https://cartes-viz.github.io/projections.html>`_ on
  top of `Cartopy <https://scitools.org.uk/cartopy/docs/latest/reference/projections.html>`_.
  Here, the Lambert93 projection is picked as it is a standard projection in France.

  .. tip::

    :ref:`How to pick a projection for a map?`

  All traffic objects which may be represented on a map are equipped with a
  :meth:`~traffic.core.mixins.ShapelyMixin.plot` method.

  .. jupyter-execute::

    from cartes.crs import Lambert93
    from traffic.data import airports

    with plt.style.context("traffic"):

        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

        airports["LFPO"].plot(ax, footprint=False, runways=dict(linewidth=1))
        for flight in subset:
            flight.plot(ax, linewidth=2)

        ax.set_title("Landing trajectories at Paris–Orly airport")


- with Altair, the initial method is
  :meth:`~traffic.core.mixins.ShapelyMixin.geoencode`

  .. jupyter-execute::

    from traffic.data import airports

    chart = (
        alt.layer(
            *(flight.geoencode().encode(alt.Color("callsign:N")) for flight in subset)
        )
        .properties(title="Landing trajectories at Paris–Orly airport")
        .configure_legend(orient="bottom")
        .configure_view(stroke=None)
        .configure_title(anchor="start", font="Lato", fontSize=16)
    )
    chart

- for quick interactive representations **with few elements**, the
  Leaflet widget is a good option:

  .. jupyter-execute::

    subset.map_leaflet(zoom=8)


Low-altitude trajectory patterns in Paris metropolitan area
-----------------------------------------------------------


The ``quickstart`` dataset contains a collection of low altitude trajectories.
In this section, we aim to display trajectory patterns of aircraft landing or
taking off from any of Paris area airport.

It is often a good practice to just plot the data as is before we get an idea of
how to proceed.


.. jupyter-execute::

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
        quickstart.plot(ax, alpha=.7)


We see here several flows converging mostly in the two major airports in Paris
(i.e., Orly ``LFPO`` and Charles-de-Gaulle ``LFPG``). However, more airports are
also visible, e.g. Beauvais airport to the North.

We can try to put a different colour to landing trajectories and take-off
trajectories to make this plot more meaningful. A first trick could be to pick a
colour based on the vertical rate average value.

.. jupyter-execute::

    import pandas as pd

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

        for flight in quickstart:
            if pd.isna(flight.vertical_rate_mean):
                continue
            if flight.vertical_rate_mean < -500:
                flight.plot(ax, color="#4c78a8", alpha=0.5)  # blue
            elif flight.vertical_rate_mean > 1000:
                flight.plot(ax, color="#f58518", alpha=0.5)  # orange
            else:
                flight.plot(ax, color="#54a24b", alpha=0.5)  # green

This approach is not perfect (there are quite some green trajectories) but gives
a good first idea of how traffic organizes itself. Let's try to focus on the
traffic to and from one airport, e.g. ``LFPO``, in order to refine the
methodology.

A first approach to select those trajectories would be to pick the first/last
point of the :class:`~traffic.core.Flight` and check whether it falls within the
geographical scope of the airport. In the following snippet, we do things a bit
differently: we check whether the first/last 5 minutes of the trajectory
intersects the shape of the airport.

.. jupyter-execute::

    from traffic.data import airports

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

        for flight in quickstart:
            if pd.isna(flight.vertical_rate_mean):
                continue
            if flight.vertical_rate_mean < -500:
                if flight.last("5 min").intersects(airports["LFPO"]):
                    flight.plot(ax, color="#4c78a8", alpha=0.5)
            elif flight.vertical_rate_mean > 1000:
                if flight.first("5 min").intersects(airports["LFPO"]):
                    flight.plot(ax, color="#f58518", alpha=0.5)

What is now becoming confusing is that there seems to have been a change in
runway configuration during the time interval covered by the dataset. It would
now probably become more comfortable if we could identify the runway used by
aircraft for take off or landing.

traffic provides :meth:`~traffic.core.Flight.aligned_on_ils` for landing and
:meth:`~traffic.core.Flight.takeoff_from_runway` for take-off. Both methods
return a :meth:`~traffic.core.FlightIterator`, so if we consider that all
trajectories have only one landing attempt on that day, we need to apply
:meth:`~traffic.core.FlightIterator.next` to get the first trajectory segment
matching, and extract relevant information (the runway information):

.. jupyter-execute::

    import pandas as pd
    from tqdm.rich import tqdm

    information = list()

    for flight in tqdm(quickstart):
        if landing := flight.aligned_on_ils("LFPO").next():
            information.append(
                {
                    "callsign": flight.callsign,
                    "icao24": flight.icao24,
                    "airport": "LFPO",
                    "stop": landing.stop,
                    "ILS": landing.ILS_max,
                }
            )
        elif landing := flight.aligned_on_ils("LFPG").next():
            information.append(
                {
                    "callsign": flight.callsign,
                    "icao24": flight.icao24,
                    "airport": "LFPG",
                    "stop": landing.stop,
                    "ILS": landing.ILS_max,
                }
            )
        elif landing := flight.aligned_on_ils("LFPB").next():
            information.append(
                {
                    "callsign": flight.callsign,
                    "icao24": flight.icao24,
                    "airport": "LFPB",
                    "stop": landing.stop,
                    "ILS": landing.ILS_max,
                }
            )


    stats = pd.DataFrame.from_records(information)
    stats


.. jupyter-execute::

    chart = (
        alt.Chart(stats)
        .encode(
            alt.X("utcdayhoursminutesseconds(stop)", axis=alt.Axis(format="%H:%M"), title=None),
            alt.Y("ILS", title=None),
            alt.Color("ILS", legend=None),
            alt.Row("airport", title=None),
        )
        .mark_square(size=80)
        .resolve_scale(y="independent")
        .configure_header(
            labelOrient="top",
            labelAnchor="start",
            labelFont="Lato",
            labelFontWeight="bold",
            labelFontSize=16,
        )
        .configure_axis(labelFontSize=13)
        .properties(width=600)
    )
    chart

It appears here that there has been a coordinated runway configuration
change around 13:20Z in all Paris airports. This suggests we should plot how
traffic organizes in both configurations.

.. jupyter-execute::

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=Lambert93()))

        for flight in quickstart:
            if segment := flight.aligned_on_ils("LFPO").next():
                index = int(flight.stop <= pd.Timestamp("2021-10-07 13:30Z"))
                flight.plot(ax[index], color="#4c78a8", alpha=0.5)
            elif segment := flight.takeoff_from_runway("LFPO").next():
                index = int(segment.start <= pd.Timestamp("2021-10-07 13:20Z"))
                flight.plot(ax[index], color="#f58518", alpha=0.5)


So it is now time to do a preliminary visualization with a basic background,
including administrative boundaries of Greater Paris Area and the Seine river as
an additional landmark:

.. jupyter-execute::

    from cartes.atlas import france
    from cartes.crs import Lambert93, PlateCarree
    from cartes.osm import Nominatim


    # background elements
    paris_area = france.data.query("ID_1 == 1000")
    seine_river = (
        Nominatim.search("Seine river, France")
        .shape.intersection(
            paris_area.union_all().buffer(0.1)
        )
    )

    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            3, 2, figsize=(10, 15), subplot_kw=dict(projection=Lambert93())
        )

        airport_codes = ["LFPO", "LFPG", "LFPB"]
        for flight in quickstart:
            phases = flight.phases()

            if phases.query('phase == "DESCENT"'):
                # Determine on which ax to plot based on detected airport
                for airport_index, airport in enumerate(airport_codes):
                    if segment := flight.aligned_on_ils(airport).next():
                        # Determine on which column to plot based on time
                        time_index = int(segment.stop <= pd.Timestamp("2021-10-07 13:20Z"))
                        flight.plot(
                            ax[airport_index, time_index], color="#4c78a8", alpha=0.4
                        )
                        break

            elif phases.query('phase == "CLIMB"'):
                # Determine on which ax to plot based on detected airport
                for airport_index, airport in enumerate(airport_codes):
                    if segment := flight.takeoff_from_runway(airport).next():
                        # Determine on which column to plot based on time
                        time_index = int(segment.start <= pd.Timestamp("2021-10-07 13:20Z"))
                        flight.plot(
                            ax[airport_index, time_index], color="#f58518", alpha=0.4
                        )
                        break

        # Annotate each map with airport information
        for i, airport in enumerate(airport_codes):
            ax[i, 0].set_title(f"{airport}", loc="left", y=0.8)

        for ax_ in ax.ravel():
            # Background map
            ax_.add_geometries(
                [seine_river], crs=PlateCarree(),
                facecolor="none", edgecolor="#9ecae9", linewidth=1.5,
            )
            paris_area.set_crs(4326).to_crs(2154).plot(
                ax=ax_,
                facecolor="none", edgecolor="#bab0ac", linestyle="dotted",
            )

            ax_.set_extent((0.78, 4.06, 47.7, 49.7))

        fig.suptitle(
            "West and East configurations in Paris airports",
            fontsize=16, x=0.1, y=0.9, ha="left",
        )

Declarative trajectory processing
---------------------------------

Basic operations on :class:`~traffic.core.Flight` objects define a specific
language which enables to express programmatically any kind of preprocessing.
The downside with programmatic preprocessing is that it may become unnecessarily
complex because of safeguards, nested loops and conditions necessary to express
even basic treatments.

The main issue with the code above is that **code for preprocessing and code for
visualization** are strongly connected: it is impossible to produce a
visualization without running “heavy” processing, as subsets of trajectories are
never stored as :class:`~traffic.core.Traffic` collections for future reuse.

There are several ways to collect trajectories:

- with trajectory arithmetic: the ``+`` operator  (and therefore the
  sum() Python built-in function) between :class:`~traffic.core.Flight` and
  :class:`~traffic.core.Traffic` objects always returns a new
  :class:`~traffic.core.Traffic` object;

- the :meth:`~traffic.core.Traffic.from_flights` class method builds a
  :class:`~traffic.core.Traffic` object from an iterable structure of
  :class:`~traffic.core.Flight` objects. It is more robust than the sum()
  Python function as it will ignore ``None`` objects which may be found in the
  iterable.

  .. jupyter-execute::

      from traffic.core import Traffic

      def select_landing(airport: "Airport"):
          for flight in quickstart:
              if low_alt := flight.query("altitude < 3000"):         # Flight -> None or Flight
                  if not pd.isna(v_mean := low_alt.vertical_rate_mean) and v_mean < -500:  # Flight -> bool
                      if low_alt.intersects(airport):                # Flight -> bool
                          if low_alt.aligned_on_ils(airport).has():  # Flight -> bool
                              yield low_alt.last("10 min")           # Flight -> None or Flight

      # Traffic.from_flights is more robust than sum() as the function may yield some None values
      Traffic.from_flights(select_landing(airports["LFPO"]))

.. tip::

    :ref:`Lazy iteration <traffic.core.lazy>` offers flattened specifications of
    trajectory preprocessing operations. Operations are stacked before being
    evaluated in a single iteration, using multiprocessing if needed, only after
    the specification is fully described.

    *Lazy evaluation* is a common wording in functional programming languages.
    It refers to a mechanism where the actual evaluation is deferred.

When you stack any :class:`~traffic.core.Flight` method returning an
``Optional[Flight]`` or a boolean, a lazy iteration is triggered. You may
remember that:

- Most :class:`~traffic.core.Flight` methods returning a ``Flight``, a boolean
  or ``None`` can be stacked on :class:`~traffic.core.Traffic` structures;
- When such a method is stacked, it is **not** evaluated, just pushed
  for later evaluation;
- The final ``.eval()`` call starts one single iteration and apply all
  stacked method to every :class:`~traffic.core.Flight` it can iterate on.
- If one of the methods returns ``False`` or ``None``, the
  :class:`~traffic.core.Flight` is discarded;
- If one of the methods returns ``True``, the :class:`~traffic.core.Flight` is
  passed as is not the next method.

The landing trajectory selection rewrites as:

.. jupyter-execute::

    (
        quickstart.query("altitude < 3000")      # Traffic -> None | Traffic
        # Lazy iteration is triggered here by the .feature_lt method
        .feature_lt("vertical_rate_mean", -500)  # Flight -> None | Flight
        .intersects(airports["LFPO"])            # Flight -> bool
        .has('aligned_on_ils("LFPO")')           # Flight -> bool
        .last("10 min")                          # Flight -> None | Flight
        # Now evaluation is triggered on 4 cores
        .eval(max_workers=4)  # the desc= argument creates a progress bar
    )

.. note::

    The :meth:`~traffic.core.Flight.aligned_on_ils` call (without considerations
    on the vertical rate and intersections) is actually enough for our needs
    here, but more methods were stacked for explanatory purposes.


For reference, look at the subtle differences between the following processing:

- take the last 10 minutes of trajectories landing at LFPO (similar to above):

    .. jupyter-execute::

        t1 = (
            quickstart
            .has("aligned_on_ils('LFPO')")
            .last('10 min')
            .eval(max_workers=4)
        )

        with plt.style.context('traffic'):
            fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
            t1.plot(ax, color="#f58518")
            airports['LFPO'].plot(
                ax, footprint=False,
                runways=dict(linewidth=1, color='black', zorder=3)
            )
            ax.spines['geo'].set_visible(False)

- take the last minute of the segment of trajectory which is aligned on runway 06:

    .. jupyter-execute::

        t2 = (
            quickstart
            .next('aligned_on_ils("LFPO")')
            .query("ILS == '06'")
            .last("1 min")
            .eval(max_workers=4)
        )

        with plt.style.context('traffic'):
            fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
            t2.plot(ax, color="#f58518")
            airports['LFPO'].plot(ax, labels=dict(fontsize=11))
            ax.spines['geo'].set_visible(False)

- select full trajectories landing on runway 06 from one minute before landing:

    .. jupyter-execute::

        import pandas as pd

        def last_minute_with_taxi(flight: "Flight") -> "None | Flight":
            for segment in flight.aligned_on_ils("LFPO"):
                if segment.ILS_max == "06":
                    return flight.after(segment.stop - pd.Timedelta("1 min"))

        t3 = quickstart.iterate_lazy().pipe(last_minute_with_taxi).eval()

        with plt.style.context('traffic'):
            fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
            t3.plot(ax, color="#f58518", zorder=3)
            airports['LFPO'].plot(ax, labels=dict(fontsize=11))
            ax.spines['geo'].set_visible(False)

- select trajectories with more than one runway alignment at LFPG:

    .. jupyter-execute::

        def more_than_one_alignment(flight: "Flight") -> "None | Flight":
            segments = flight.aligned_on_ils("LFPG")
            if first := next(segments, None):
                if second := next(segments, None):
                    return flight.after(first.start - pd.Timedelta('90s'))

        t4 = quickstart.iterate_lazy().pipe(more_than_one_alignment).eval()

        flight = t4[0]
        segments = flight.aligned_on_ils("LFPG")
        first = next(segments)
        forward = first.first("70s").forward(minutes=4)

        chart = (
            alt.layer(
                airports["LFPG"].geoencode(
                    footprint=False,
                    runways=dict(strokeWidth=1),
                    labels=dict(fontSize=10),
                ),
                flight.geoencode().mark_line(stroke="#bab0ac"),
                forward.geoencode(stroke="#79706e", strokeDash=[7, 3], strokeWidth=0.8),
                first.geoencode().encode(alt.Color("ILS")),
                next(segments).geoencode().encode(alt.Color("ILS")),
            )
            .properties(
                title=f"Runway change at LFPG airport with {flight.callsign}",
                width=600,
            )
            .configure_view(stroke=None)
            .configure_legend(orient="bottom")
            .configure_title(font="Lato", fontSize=16, anchor="start")
        )
        chart
