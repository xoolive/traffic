How to estimate the fuel burnt by an aircraft?
==============================================

The traffic library integrates the `OpenAP <https://openap.dev/>`__ aircraft
performance and emission model.

We can demonstrate its use with an example flight extracted from a real flight
data recorder of an Airbus A320-216 aircraft. Data is anonymised, so there is no
geographical information about the trajectory and timestamps are just wrong.


.. admonition:: About anonymisation

    - I may consider adding extra features to this dataset if it could help
      validate or illustrate a use case with the library. The bottom line is
      that anonymisation (tail number) should not be obviously broken.

    - If you find a way to break the anonymisation of this particular flight,
      good for you: nobody will confirm or deny your claim.

    - You may manage to find the city-pair associated with that flight: if you
      reconstruct the whole trajectory with a reasonable process, let's write a
      tutorial page to illustrate it.

.. jupyter-execute::

    from traffic.data.samples import fuelflow_a320

There are enough features (and some more) in the provided data to estimate the
fuel flow with OpenAP and compare the result with real data.

.. jupyter-execute::

    fuelflow_a320.data[['timestamp', 'altitude', 'groundspeed', 'CAS', 'vertical_acceleration', 'weight', 'fuelflow']]

User interface
--------------

The most direct use of the API, is with the :meth:`~traffic.core.Flight.fuelflow`
method:

.. automethod:: traffic.core.Flight.fuelflow


In this dataset:

- | the *vertical rate* is not available as it is not directly measured on
    aircraft. Here, we consider the most simple approach and derive it from the
    altitude.
  | In practice, the *vertical acceleration* is used to filter the vertical rate
    signal. It has been made available in this dataset (as a g-force)

- the *fuel flow* (the ground truth) is provided in kg/h, we convert it in kg/s
  for compatibility reason with the output of the OpenAP interface.

.. jupyter-execute::

    f = fuelflow_a320.assign(
        # the vertical_rate is not present in the data
        vertical_rate=lambda df: df.altitude.diff().fillna(0) * 60,
        # convert to kg/s
        fuelflow=lambda df: df.fuelflow / 3600,
    )

Sources of uncertainty
----------------------

Let's analyse the results produced with the following runs.

.. jupyter-execute::

    import altair as alt
    alt.data_transformers.disable_max_rows()


    def plot_flow(flight):
        return flight.chart().encode(
            alt.X(
                "utchoursminutesseconds(timestamp)",
                axis=alt.Axis(title=None, format="%H:%M"),
            ),
            alt.Y("fuelflow", axis=alt.Axis(title="fuel flow (in kg/s)")),
            alt.Color("legend", title=None),
        )


    def chart_flow(*flights):
        return (
            alt.layer(*(plot_flow(flight) for flight in flights))
            .properties(width=600, height=250)
            .configure_axis(
                labelFontSize=14, titleFontSize=16,
                titleAngle=0, titleY=-12, titleAnchor="start",
            )
            .configure_legend(
                orient="bottom", columns=1,
                labelFontSize=14, symbolSize=400, symbolStrokeWidth=3,
            )
        )

Default parameters
~~~~~~~~~~~~~~~~~~

The default approach considers the default engine (which is the correct one for
this particular aircraft), assumes the initial mass of the aircraft to be 90% of
the initial take-off mass, and computes the TAS based on the available CAS.

The ``typecode="A320"`` must be passed as a parameter because the ``icao24``
parameter is not provided in this example.

.. jupyter-execute::

    resampled = f.resample("5s")
    openap = resampled.fuelflow(typecode="A320")

    chart_flow(
        openap.assign(legend="OpenAP estimation"),
        resampled.assign(legend="Real fuelflow")
    )

.. jupyter-execute::

    real_fuel = resampled.weight_max - resampled.weight_min
    estimated_fuel = openap.fuel_max

    print(f"Total burnt fuel: {real_fuel:.0f}kg, OpenAP estimation: {estimated_fuel:.0f}kg")
    print(f"Error: {abs(estimated_fuel - real_fuel) / real_fuel:.0%}")

Impact of the take-off mass
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the weight of the aircraft is available along this particular
trajectory---note that this is most likely an approximation too, based on the
quantity of fuel loaded, the estimation of fuel burnt, cargo, number of embarked
passengers, etc.---we can see that a better estimation of the mass slightly
improves the estimation of the fuel flow.

.. jupyter-execute::

    resampled = f.resample("5s")
    openap = resampled.fuelflow(typecode="A320", initial_mass=resampled.weight_max)

    chart_flow(
        openap.assign(legend="OpenAP estimation"),
        resampled.assign(legend="Real fuelflow")
    )

.. jupyter-execute::

    real_fuel = resampled.weight_max - resampled.weight_min
    estimated_fuel = openap.fuel_max

    print(f"Total burnt fuel: {real_fuel:.0f}kg, OpenAP estimation: {estimated_fuel:.0f}kg")
    print(f"Error: {abs(estimated_fuel - real_fuel) / real_fuel:.0%}")

Impact of the wind
~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    resampled = f.resample("5s").drop(columns=["CAS"])
    openap = resampled.fuelflow(typecode="A320")

    chart_flow(
        openap.assign(legend="OpenAP estimation"),
        resampled.assign(legend="Real fuelflow")
    )

.. jupyter-execute::

    real_fuel = resampled.weight_max - resampled.weight_min
    estimated_fuel = openap.fuel_max

    print(f"Total burnt fuel: {real_fuel:.0f}kg, OpenAP estimation: {estimated_fuel:.0f}kg")
    print(f"Error: {abs(estimated_fuel - real_fuel) / real_fuel:.0%}")

There are two ways to take wind into account when estimating fuel flow based on ADS-B data:

- use information from extended Mode S in areas of the world where it is
  available (see :meth:`~traffic.core.Traffic.query_ehs`);
- interpolate wind from GRIB files provided by Meteorological Agencies and use
  the information to compute the true air speed (TAS)

Influence of the sampling rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current Python implementation of fuel flow estimation is a bit slow, but
changing the sampling rate of the trajectories in order to accelerate processing
seems to have little impact on the final estimation.


.. jupyter-execute::

    resampled = f.resample("20s")
    openap = resampled.fuelflow(typecode="A320")

    chart_flow(
        openap.assign(legend="OpenAP estimation"),
        resampled.assign(legend="Real fuelflow")
    )

.. jupyter-execute::

    real_fuel = resampled.weight_max - resampled.weight_min
    estimated_fuel = openap.fuel_max

    print(f"Total burnt fuel: {real_fuel:.0f}kg, OpenAP estimation: {estimated_fuel:.0f}kg")
    print(f"Error: {abs(estimated_fuel - real_fuel) / real_fuel:.0%}")

Influence of the engine type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine type has a serious impact on the fuel flow estimation even if the
general trend looks similar. If you know the engine type for each aircraft, it
may be more reasonable to specify it when running your estimation.

.. jupyter-execute::

    resampled = f.resample("5s")
    openap = resampled.fuelflow(typecode="A320", engine="CFM56-5B5")  # default/correct is CFM56-5B4

    chart_flow(
        openap.assign(legend="OpenAP estimation"),
        resampled.assign(legend="Real fuelflow")
    )

.. jupyter-execute::

    real_fuel = resampled.weight_max - resampled.weight_min
    estimated_fuel = openap.fuel_max

    print(f"Total burnt fuel: {real_fuel:.0f}kg, OpenAP estimation: {estimated_fuel:.0f}kg")
    print(f"Error: {abs(estimated_fuel - real_fuel) / real_fuel:.0%}")
