Aircraft
--------

The aircraft database by `junzis <https://junzisun.com/adb/>`__ is
available through the library.

Basic requests are available. Most importantly, you can enrich a Traffic
structure with aircraft models.

.. code:: python

    from traffic.data import aircraft
    aircraft.operator('Air France').head()




.. raw:: html

    <div>
    <table border="0" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>icao</th>
          <th>regid</th>
          <th>mdl</th>
          <th>type</th>
          <th>operator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>3032</th>
          <td>391558</td>
          <td>F-GFKY</td>
          <td>A320</td>
          <td>Airbus A320-211</td>
          <td>Air France</td>
        </tr>
        <tr>
          <th>3418</th>
          <td>391e09</td>
          <td>F-GHQJ</td>
          <td>A320</td>
          <td>Airbus A320-211</td>
          <td>Air France</td>
        </tr>
        <tr>
          <th>3419</th>
          <td>391e0b</td>
          <td>F-GHQL</td>
          <td>A320</td>
          <td>Airbus A320-211</td>
          <td>Air France</td>
        </tr>
        <tr>
          <th>3420</th>
          <td>391e0c</td>
          <td>F-GHQM</td>
          <td>A320</td>
          <td>Airbus A320-211</td>
          <td>Air France</td>
        </tr>
        <tr>
          <th>3446</th>
          <td>392263</td>
          <td>F-GITD</td>
          <td>B744</td>
          <td>Boeing 747-428</td>
          <td>Air France</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    aircraft['F-GFKY']




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="0" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>icao</th>
          <th>regid</th>
          <th>mdl</th>
          <th>type</th>
          <th>operator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>3032</th>
          <td>391558</td>
          <td>F-GFKY</td>
          <td>A320</td>
          <td>Airbus A320-211</td>
          <td>Air France</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    aircraft['391558']




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="0" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>icao</th>
          <th>regid</th>
          <th>mdl</th>
          <th>type</th>
          <th>operator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>3032</th>
          <td>391558</td>
          <td>F-GFKY</td>
          <td>A320</td>
          <td>Airbus A320-211</td>
          <td>Air France</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    from traffic.core import Traffic
    t = Traffic.from_file("../data/sample_opensky.pkl")
    t_ext = aircraft.merge(t)
    t_ext['AFR23FK'].at()

.. parsed-literal::
    alert                                 False
    altitude                                375
    callsign                            AFR23FK
    geoaltitude                             550
    groundspeed                         147.611
    hour                             1500235200
    icao24                               393322
    last_position    2017-07-16 22:14:35.733000
    latitude                            43.6282
    longitude                           1.36716
    onground                               True
    spi                                   False
    squawk                                 1000
    timestamp               2017-07-16 22:19:35
    track                               322.431
    vertical_rate                          -576
    regid                                F-GMZC
    mdl                                    A321
    type                        Airbus A321-111
    operator                         Air France
    Name: 329480, dtype: object

