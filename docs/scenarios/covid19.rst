Impact of COVID-19 on worldwide aviation
----------------------------------------

.. warning::

    The link to an enriched version of the dataset will be published on a public server after few glitches are fixed. Many thanks to Martin and Jannis from The OpenSky Network.

The pandemic of coronavirus is having a serious impact on aviation around the world. The slowdown appears on data, with some regional peculiarities. 
The underlying data is currently updated daily.

Flight evolution per airport
============================

The following plot displays the current trend in number of departing aircraft from airports in various areas around the world (covered by The OpenSky Network).

.. raw:: html

    <div id="covid19_airports"></div>

    <script type="text/javascript">
      var spec = "../_static/covid19_airports.json";
      vegaEmbed('#covid19_airports', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>

Currently, the trend shows:

- a slow decrease from February in Asian airports (an early one in Hong-Kong);
- European airports plummetting since early day of March;
- America started dropping later;
- India almost stopped all traffic (VABB, VIDP).

Flight evolution per airline
============================

.. raw:: html

    <div id="covid19_airlines"></div>

    <script type="text/javascript">
      var spec = "../_static/covid19_airlines.json";
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

On the `Impala shell <../opensky_impala.html>`_, a particular table shows flight lists with associated origin and destination airport.

.. code:: python

    flightlist = (
        opensky
        # the request, adjust dates
        .flightlist("2020-01-01", "2020-04-01")
        # some basic sanity checking
        .query("callsign == callsign and firstseen == firstseen")
    )

We will select a few subset of airports for visualisation and build a specific table limited to these airports: the idea is to plot the number of departing aircraft per day for each of the following airports.

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
    