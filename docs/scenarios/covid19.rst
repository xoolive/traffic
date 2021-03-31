Impact of COVID-19 on worldwide aviation
----------------------------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3928550.svg
   :target: https://doi.org/10.5281/zenodo.3928550

.. admonition:: Information

    The dataset constructed for the analysis on this page is now `available online <https://opensky-network.org/datasets/covid-19/>`_, with a proper introduction on the `OpenSky Network blog <https://opensky-network.org/community/blog/item/6-opensky-covid-19-flight-dataset>`_.
    Many thanks to the whole team.

The pandemic of coronavirus is having a serious impact on aviation around the world. The slowdown appears on data, with some regional peculiarities. 
The underlying data is currently updated daily.

Similar initiatives to analyse the impact of COVID-19 on aviation are available here:

- Eurocontrol PRU analysis on their `website <https://ansperformance.eu/covid/>`_, with downloadable data
- beautiful maps by the `Washington Post <https://www.washingtonpost.com/graphics/2020/business/coronavirus-airline-industry-collapse/>`_
- FlightRadar24 view on their `blog <https://www.flightradar24.com/blog/tracking-marchs-historic-drop-in-air-traffic/>`_
- the quarterly `Monetary Policy Report <https://www.bankofengland.co.uk/report/2020/monetary-policy-report-financial-stability-report-may-2020>`_ by the Bank of England uses OpenSky data as well, but the one obtained through the `live API </opensky_rest.html>`_
- experiment of data visualisations by `Craig Taylor <https://twitter.com/CraigTaylorViz/status/1258083226549194753>`_

With the same dataset, we found:

- a focus on American airports by `@ethanklapper <https://twitter.com/ethanklapper/status/1246167346693144578>`_ on Twitter
- an excellent Observable notebook by `@lounjukk <https://observablehq.com/@lounjukk/flights-during-covid-19-pandemic>`_ replaying all the traffic on a world map
- `Baptiste Coulmont <http://coulmont.com/blog/2020/05/04/dataconfinement1/>`_ analyses the impact of the lockdown in France based on various sources of open data, including this dataset (in French)
- French newspaper `Liberation <https://www.liberation.fr/apps/2020/05/bilan-confinement/>`_ did a similar job few days later, with more data visualisations 
- `How COVID-19 has rocked US domestic flights <https://evandenmark.github.io/ForSpaciousSkies/>`_ by `@EvDenmark <https://twitter.com/EvDenmark/status/1260922351732101120>`_
- `How have SARS-CoV-2's limitations impacted global aviation traffic? <https://sirbenedick.github.io/corona-aviation-impact/>`_ by `@sirbenedick <https://github.com/SirBenedick>`_
- `Visualization of Air Traffic during Covid-19 Pandemic <https://towardsdatascience.com/visualization-of-air-traffic-during-covid-19-pandemic-c5941b049401>`_ 
- `What Explains Temporal and Geographic Variation in the Early US Coronavirus Pandemic? <https://www.nber.org/papers/w27965>`_, DOI: `10.3386/w27965 <https://doi.org/10.3386/w27965>`_

This list may not be exhaustive. You may `open an issue <https://github.com/xoolive/traffic/issues/new>`_ on the github of the library in order to reference more websites.

Flight evolution per airport
============================

The following plot displays the current trend in number of departing aircraft from airports in various areas around the world (covered by The OpenSky Network).

.. raw:: html

    <div id="covid19_airports"></div>

    <script type="text/javascript">
      var spec = "https://raw.githubusercontent.com/traffic-viz/traffic_static/master/json/covid19_airports.json";
      vegaEmbed('#covid19_airports', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>

In earlier days, the trend showed:

- a slow decrease from February in Asian airports (an early one in Hong-Kong);
- European airports plummetting since early day of March;
- America started dropping later;
- India almost stopped all traffic (VABB, VIDP).

In the past few days, some airports and airlines seem to have experienced a slight increase of activity.

Flight evolution per airline
============================

.. raw:: html

    <div id="covid19_airlines"></div>

    <script type="text/javascript">
      var spec = "https://raw.githubusercontent.com/traffic-viz/traffic_static/master/json/covid19_airlines.json";
      vegaEmbed('#covid19_airlines', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>

Currently, the trend shows:

- decreasing patterns for regular airlines depending on the geography;
- more low-cost airlines stopping their activity;
- cargo airlines carrying on (see activities by `@simon_sat <https://twitter.com/simon_sat/status/1244643841447247872>`_ on Twitter)

Data collection and preparation
===============================

On the `Impala shell <../opensky_impala.html>`_, a particular table contains flight lists with associated origin and destination airport. The data has been downloaded using the `opensky.flightlist <https://traffic-viz.github.io/opensky_impala.html#traffic.data.adsb.opensky_impala.Impala.flightlist>`_ method, curated, then aggregated with aircraft and flight number information before being published `here <https://opensky-network.org/datasets/covid-19/>`_. Download the data and run the following:

.. code:: python

    from pathlib import Path
    import pandas as pd

    flightlist = pd.concat(
        pd.read_csv(file, parse_dates=["firstseen", "lastseen", "day"])
        for file in Path("path/to/folder").glob("flightlist_*.csv.gz")
    )

We will select a few subset of airports for visualisation and build a specific table limited to these airports: the idea is to plot the number of departing aircraft per day for each of the following airports. The plot for airlines goes along the same idea.

.. code:: python

    from traffic.data import airports
    import altair as alt

    airports_subset = [
        # Europe
        ["LFPG", "EGLL", "EHAM", "EDDF", "LEMD", "LIRF", "LSZH", "UUEE"],
        # Eastern Asia
        ["VHHH", "RJBB", "RJTT", "RKSI", "RCTP", "RPLL"],
        # Asia (other)
        ["YSSY", "YMML", "OMDB", "VABB", "VIDP", "WSSS"],
        # Americas
        ["CYYZ", "KSFO", "KLAX", "KATL", "KJFK", "SBGR"],
    ]

    data = pd.concat(
        (
            flightlist.query(f'origin == "{airport}"')
            # count the number of departing aircraft per day
            .groupby("day")
            .agg(dict(callsign="count"))
            # label the current chunk with the name of the airport
            .rename(columns=dict(callsign=airport))
            # iterate on all airports in the list hereabove
            for airport in sum(airports_subset, [])
        ),
        axis=1,
    )

    chart = alt.Chart(
        data.reset_index()
        # prepare data for altair
        .melt("day", var_name="airport", value_name="count")
        # include the name of the city associated with the airport code
        .merge(
            airports.data[["icao", "municipality"]],
            left_on="airport",
            right_on="icao",
            how="left",
        )[["day", "airport", "count", "municipality"]]
        # rename this feature 'city'
        .rename(columns=dict(municipality="city"))
    )


    def full_chart(source, subset, subset_name):

        # We have many airports, only pick a subset
        chart = source.transform_filter(
            alt.FieldOneOfPredicate(field="airport", oneOf=subset)
        )

        # When we come close to a line, highlight it
        highlight = alt.selection(
            type="single", nearest=True, on="mouseover", fields=["airport"]
        )

        # The scatter plot
        points = (
            chart.mark_point()
            .encode(
                x="day",
                y=alt.Y("count", title="# of departing flights"),
                color=alt.Color("airport", legend=alt.Legend(title=subset_name)),
                # add some legend next to  point
                tooltip=["day", "airport", "city", "count"],
                # not too noisy please
                opacity=alt.value(0.5),
            )
            .add_selection(highlight)
        )

        # The trend plot
        lines = (
            chart.mark_line()
            .encode(
                x="day",
                y="count",
                color="airport",
                size=alt.condition(~highlight, alt.value(1), alt.value(3)),
            )
            # the cloud is a bit messy, draw a trend through it
            .transform_loess("day", "count", groupby=["airport"], bandwidth=0.2)
        )

        return lines + points


    # Concatenate several plots
    result = alt.vconcat(
        *[
            full_chart(chart, airport_, subset_name).properties(width=600, height=150)
            for subset_name, airport_ in zip(
                [
                    "European airports",
                    "East-Asian airports",
                    "Asian/Australian airports",
                    "American airports",
                ],
                airports_subset,
            )
        ]
    ).resolve_scale(color="independent")

    result
    