.. raw:: html
    :file: ../embed_widgets/windfield.html

A Mode S based wind field
-------------------------

ADS-B data broadcasted by aircraft contains information about groundspeed and
true track angle of its trajectory. When proper requests are sent by a Secondary
surveillance radar, aircraft also send more information with true airspeed and
heading angle in specific BDS5,0 and BDS6,0 messages (see `Junzi Sun's website
<https://mode-s.org/decode/>`__)

Groundspeeds and true track angles are derived from the GNSS positions whereas
true airspeed is computed with traditional onboard instruments like `Pitot tubes
<https://en.wikipedia.org/wiki/Pitot_tube>`__.

Here, aircraft behave as a distributed network of moving sensors and researchers
[1]_ [2]_ have been recommending methods to derive wind fields from Mode S data.
The following method is a very basic approach to compute wind.

Please note I am not a meteorology specialist, so feel free to improve this page
where it should.

.. raw:: html

   <script type="application/vnd.jupyter.widget-view+json">
   {
       "version_major": 2,
       "version_minor": 0,
       "model_id": "c2a6efbadb4442dcbe468a3407ee237b"
   }
   </script>

   <br/>


The example above is wind averaged between 25°W and 55°E and between 32°N and
65°N, from FL200 and above on February 23th 2019, between 14:00 and 16:30 UTC.

.. code:: python

    from traffic.core import Traffic

    t = Traffic.from_file("<fill here>")

    t_extended = (
        traffic
        # download and decode EHS messages (DF 20/21)
        .query_ehs()
        # resample/interpolate one sample per second
        .resample("1s")
        # median filters
        .filter(altitude=23, track=53, heading=53, groundspeed=53, TAS=53)
        # wind triangle computation
        .compute_wind()
        # median filter
        .filter(wind_u=53, wind_v=53)
        # resample one sample per minute
        .resample("1T")
        # do not use multiprocessing to avoid denial of service
        .eval(desc="preprocessing")
    )

    # t_extended.to_pickle("wind_backup.pkl")

The result of this computation is a set of trajectories: each aircraft yields
one point per minute with a 4D-position (timestamp, latitude, longitude,
altitude) and a wind vector decomposed along a zonal speed (`wind_u`) and a
meridional speed (`wind_v`).

We then use `ipyleaflet <http://ipyleaflet.readthedocs.io/>`__ to display the
wind field. The `Velocity` widget requires two 2D matrices of zonal and
meridional components of the wind. The following method rounds lat/lon
coordinates to the closest integer and average wind in each resulting cell.

.. code:: python

    import xarray as xr

    def compute_grid(traffic: Traffic) -> xr.Dataset:

        avg = (
            traffic
            # remove NaN values, just in case
            .query("wind_u == wind_u")
            # prepare coordinates for the 4d-grid, also remove NaN in wind
            .assign(
                # round coordinates to the closest .33 latitude/longitude
                lat_=lambda df: (3 * df.latitude.round(0)) / 3,
                lon_=lambda df: (3 * df.longitude.round(0)) / 3,
                # This basic version averages on all altitudes/timeranges
                # but it is easy to use the following fields to display
                # wind fields in particular time ranges and altitude levels.
                alt_=lambda df: (3e-3 * df.altitude).round(0) / 3e-3,
                hour=lambda df: df.timestamp.dt.round("h"),
            )
            # compute the average wind
            .data[["wind_u", "wind_v", "lat_", "lon_"]]
            .groupby(["lat_", "lon_"])
            .mean()
        )

        # Unstack then fill the holes where possible (2D interpolation)
        u = avg[["wind_u"]].unstack().interpolate().values
        v = avg[["wind_v"]].unstack().interpolate().values

        return xr.Dataset(
            data_vars={
                "u_wind": xr.DataArray(u, coords=avg.index.levels),
                "v_wind": xr.DataArray(v, coords=avg.index.levels),
            }
        )


The following is a basic rendering delegated to ipyleaflet library.

.. code:: python

    from ipyleaflet import Map, Velocity, basemaps

    # t_extended = Traffic.from_file("wind_backup.pkl")

    map_ = Map(
        center=(52, 15),
        zoom=4,
        interpolation="nearest",
        basemap=basemaps.CartoDB.DarkMatter,
    )

    wind = Velocity(
        data=compute_grid(t_extended),
        zonal_speed="u_wind",
        meridional_speed="v_wind",
        latitude_dimension="lat_",
        longitude_dimension="lon_",
        velocity_scale=0.002,
        max_velocity=150,
    )

    map_.add_layer(wind)

    map_


.. [1] | Hurter, C., R. Alligier, D. Gianazza, S. Puechmorel, G. Andrienko, and N. Andrienko.
       | « Wind Parameters Extraction from Aircraft Trajectories ». Computers, Environment and Urban Systems 47 (2014): 28‑43.
       | https://doi.org/10.1016/j.compenvurbsys.2014.01.005.

.. [2] | Sun, Junzi, Huy Vu, Joost Ellerbroek, and Jacco Hoekstra.
       | « Ground-Based Wind Field Construction from Mode-S and ADS-B Data with a Novel Gas Particle Model », 2017, 9.
