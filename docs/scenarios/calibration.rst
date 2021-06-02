.. raw:: html
    :file: ../embed_widgets/leaflet.html

.. raw:: html

    <!-- The inclusion above triggers the inclusion of leaflet without conflicting other pages. -->

    <script src="../_static/calibration.geojson"></script>

Calibration flights for VOR and ILS systems
-------------------------------------------

*What is this plane doing?*

This was my first reaction after hitting on the following trajectory during an
analysis of approaches at Toulouse airport. After exchanges with ATC people, I
learned that these trajectories are flown by small aircraft working at
calibrating landing assistance systems, including ILS and VOR.

These trajectories mostly consist of many low passes over an airport and large
circles or arcs of circle. A small sample of such trajectories is included in
traffic.data.samples, and the following snippet of code will let you explore
those before an attempt of explanation.


.. raw:: html

    Select a different area/airport: <select
        id="city_selector"
        style="-webkit-appearance: none; -moz-appearance: none; -o-appearance: none; appearance: none; padding-top: 0.25em; padding-bottom: 0.25em; background-image: url('data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjIuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxOCAxOCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTggMTg7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbDpub25lO30KPC9zdHlsZT4KPHBhdGggZD0iTTUuMiw1LjlMOSw5LjdsMy44LTMuOGwxLjIsMS4ybC00LjksNWwtNC45LTVMNS4yLDUuOXoiLz4KPHBhdGggY2xhc3M9InN0MCIgZD0iTTAtMC42aDE4djE4SDBWLTAuNnoiLz4KPC9zdmc+Cg'); background-position-x: 100%; background-position-y: 50%; background-repeat: no-repeat; background-size: 20px auto; box-shadow: none; box-sizing: border-box; border-radius: 0px 0px 0px 0px; height: 28px; width: 15em; line-height: 16px; font-size: 13px;"
        onchange="city_select()"
        >
        <option value="Ajaccio, France">Ajaccio,&nbsp;France</option>
        <option value="Bornholm, Denmark">Bornholm,&nbsp;Denmark</option>
        <option value="Brussels, Belgium">Brussels,&nbsp;Belgium</option>
        <option value="Cardiff, United Kingdom">Cardiff,&nbsp;United&nbsp;Kingdom</option>
        <option value="Funchal, Madeira">Funchal,&nbsp;Madeira</option>
        <option value="Guatemala City">Guatemala&nbsp;City</option>
        <option value="Kingston, Jamaica">Kingston,&nbsp;Jamaica</option>
        <option value="Kiruna, Sweden">Kiruna,&nbsp;Sweden</option>
        <option value="Kota Kinabalu, Malaysia">Kota&nbsp;Kinabalu,&nbsp;Malaysia</option>
        <option value="Lisbon, Portugal">Lisbon,&nbsp;Portugal</option>
        <option value="London, United Kingdom">London,&nbsp;United Kingdom</option>
        <option value="Monastir, Tunisia">Monastir,&nbsp;Tunisia</option>
        <option value="Montréal, Canada">Montréal,&nbsp;Canada</option>
        <option value="Munich, Germany">Munich,&nbsp;Germany</option>
        <option value="Nouméa, New Caledonia">Nouméa,&nbsp;New&nbsp;Caledonia</option>
        <option value="Perth, Australia">Perth,&nbsp;Australia</option>
        <option value="Toulouse, France" selected='selected'>Toulouse,&nbsp;France</option>
        <option value="Vancouver, Canada">Vancouver,&nbsp;Canada</option>
        <option value="Vienna, Austria">Vienna,&nbsp;Austria</option>
    </select>

    <div id="mymap" style="width: 100%; height: 500px; margin-bottom: 1em; margin-top: 1em"></div>

    <script>
        var mymap;
        var mymap_ready = false;

        function city_select() {
            var inp = document.getElementById('city_selector');
            var osm_url = 'https://nominatim.openstreetmap.org/search?format=json&limit=1&q='
            $.getJSON(
                osm_url + inp.value, function(data) {
                    $.each(
                        data, function(key, val) {
                            mymap.setView([val.lat, val.lon], 9);
                        }
                    )
                }
            )
        };

        setInterval(function() {

        if ((typeof (L) !== "undefined") & ~mymap_ready) {

        mymap = L.map( 'mymap', {
            center: [43.59601301626894, 1.4321018748075245],
            scrollWheelZoom: false,
            zoom: 9
        })

        L.tileLayer('http://stamen-tiles-a.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution:
            'Map tiles by <a href="http://stamen.com/">Stamen Design</a>, '+
            'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '+
            'Data by <a href="http://openstreetmap.org/">OpenStreetMap</a>, '+
            'under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.',
            id: 'stamen.terrain'
        }).addTo(mymap);


        L.geoJson(
            calibration_trajectories,
            style={
                color: '#0033FF'
            }
        ).bindPopup(function (layer) {
            return "<b>Trajectory " 
              + layer.feature.properties.flight_id
              + "</b><br/> with aircraft "
              + layer.feature.properties.registration
              + " and callsign "
              + layer.feature.properties.callsign;
        }).addTo(mymap);

        mymap_ready = true

      }}, 1000)
    </script>

You may click on trajectories for more information.

A short introduction to radio-navigation
========================================

A number of navigational systems emerged in the second half of the 20th century,
mainly based on VHF communications. Some of them were designed for surveillance,
like `Secondary surveillance radars
<https://en.wikipedia.org/wiki/Secondary_surveillance_radar>`_; other systems
also came to assist pilot in navigation and landing.


`VOR <https://en.wikipedia.org/wiki/VHF_omnidirectional_range>`_ (VHF
Omnidirectional Range) ground stations send an omnidirectional master signal
(on a predefined frequency determined for each station) and a second highly
directional signal. Aircraft measure the phase difference between the two
signals, which corresponds to the bearing from the station to the aircraft. [1]_

Historically, airways were laid out between VORs, which stand at the
intersections between those routes. Advances in GNSS (understand GPS) make these
stations less necessary.

VOR stations often host a `DME
<https://en.wikipedia.org/wiki/Distance_measuring_equipment>`_ (Distance
Measuring Equipment). The principle is similar to radar ranging, except the
roles of the aircraft and of the ground station are reversed. The aircraft sends
a signal to the DME; the DME repeats the same signal 50 μs after reception. When
the aircraft receives a copy of the sent messages, it measures the time of
travel to the DME, subtracts 50 μs and divides the results by 2: speed of light
gives an estimation of the distance between the aircraft to the ground station.

`ILS <https://en.wikipedia.org/wiki/Instrument_landing_system>`_ (Instrument
Landing Systems) consists of two guidance systems: a lateral one (the LOC, for
*localizer*) and a vertical one (the GS, for *glide slope*, also *glide path*).
The localizer usually consists of several pairs of directional antennas placed
beyond the departure end of the runway. [2]_

Local authorities define very strict thresholds for accuracy: internal
monitoring shall switch off the system if the accuracy of the signal is not
appropriate. All radio-navigation beacons (including VOR, DME and ILS) are
checked periodically by specially equipped aircraft. In particular, the VOR test
consists of flying around the beacon in circles at defined distances and along
several radials.

.. [1] Decoding VOR signals can be a fun exercice for the amateur software radio developper. (`link <https://www.radiojitter.com/real-time-decoding-of-vor-using-rtl-sdr/>`_)
.. [2] Check for them next time you drive around an airport!


A basic analysis of VOR calibration trajectories
================================================

We can have a look at the first trajectory in the calibration dataset. The
aircraft takes off from Ajaccio airport before flying concentric circles and
radials. There must be a VOR around, we can search in the navaid database:

.. code:: python

   # see https://traffic-viz.github.io/samples.html if any issue on import
   from traffic.data.samples.calibration import ajaccio
   from traffic.data import navaids

   navaids.extent(ajaccio).query('type == "VOR"')

.. raw:: html

   <table class="dataframe" border="0">
     <thead>
       <tr style="text-align: right;">
         <th></th>
         <th>name</th>
         <th>type</th>
         <th>latitude</th>
         <th>longitude</th>
         <th>altitude</th>
         <th>frequency</th>
         <th>description</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <th>126858</th>
         <td>AJO</td>
         <td>VOR</td>
         <td>41.770528</td>
         <td>8.774667</td>
         <td>2142.0</td>
         <td>114.8</td>
         <td>AJACCIO VOR-DME</td>
       </tr>
       <tr>
         <th>127828</th>
         <td>FGI</td>
         <td>VOR</td>
         <td>41.502194</td>
         <td>9.083417</td>
         <td>87.0</td>
         <td>116.7</td>
         <td>FIGARI VOR-DME</td>
       </tr>
     </tbody>
   </table>


Next step is to compute for each point the distance and bearing from the VOR to
each point of the trajectory. The parts of the trajectory that are of interest
are the ones with little to no variation in the distance (circles) and in the
bearing (radials) to the VOR.

.. code:: python

   vor = navaids.extent(ajaccio)['AJO']

   ajaccio = (
       ajaccio.distance(vor)  # add a distance column (in nm) w.r.t the VOR
       .bearing(vor)  # add a bearing column w.r.t the VOR
       .assign(
           distance_diff=lambda df: df.distance.diff().abs(),  # large circles
           bearing_diff=lambda df: df.bearing.diff().abs(),  # long radials
       )
   )

We can write a simple .query() followed by a .split() method to select all
segments with a constant bearing with respect to the selected VOR.

.. code:: python

    for segment in ajaccio.query('bearing_diff < .01').split('1T'):
        if segment.longer_than('5 minutes'):
            print(segment.duration)

    # 0 days 00:05:05
    # 0 days 00:05:10
    # 0 days 00:17:20
    # 0 days 00:22:35
    # 0 days 00:05:20
    # 0 days 00:09:40
    # 0 days 00:08:15


We have all we need to enhance the interesting parts of the trajectory now:

.. code:: python

   %matplotlib inline

   import matplotlib.pyplot as plt
   import pandas as pd

   from traffic.drawing import Lambert93, countries
   from traffic.data import airports

   point_params = dict(zorder=5, text_kw=dict(fontname="Ubuntu", fontsize=15))
   box_params = dict(boxstyle="round", facecolor="lightpink", alpha=.7, zorder=5)

   with plt.style.context("traffic"):

       fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

       ax.add_feature(countries(edgecolor="midnightblue"))

       airports["LFKJ"].point.plot(ax, marker="^", **point_params)
       shift_vor = dict(units="dots", x=20, y=10)
       vor.plot(ax, marker="h", shift=shift_vor, **point_params)

       # background with the full trajectory
       ajaccio.plot(ax, color="#aaaaaa", linestyle="--")

       # plot large circles in red
       for segment in ajaccio.query("distance_diff < .02").split("1 minute"):
           # only print the segment if it is long enough
           if segment.longer_than("3 minutes"):
               segment.plot(ax, color="crimson")
               distance_vor = segment.data.distance.mean()

               # an annotation with the radius of the circle
               segment.at().plot(
                   ax, alpha=0,  # We don't need the point, only the text
                   text_kw=dict(s=f"{distance_vor:.1f} nm", bbox=box_params)
               )

       for segment in ajaccio.query("bearing_diff < .01").split("1 minute"):
           # only print the segment if it is long enough
           if segment.longer_than("3 minutes"):
               segment.plot(ax, color="forestgreen")

       ax.set_extent((7.6, 9.9, 41.3, 43.3))
       ax.spines['geo'].set_visible(False)
       ax.background_patch.set_visible(False)

.. image:: images/ajaccio_map.png
   :scale: 70%
   :alt: Situational map
   :align: center

The following map displays the result of a similar processing on the other VOR
calibration trajectories from the sample dataset. [3]_


.. raw:: html

    Select a different VOR: <select
        id="vor_selector"
        style="-webkit-appearance: none; -moz-appearance: none; -o-appearance: none; appearance: none; padding-top: 0.25em; padding-bottom: 0.25em; background-image: url('data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjIuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxOCAxOCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTggMTg7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbDpub25lO30KPC9zdHlsZT4KPHBhdGggZD0iTTUuMiw1LjlMOSw5LjdsMy44LTMuOGwxLjIsMS4ybC00LjksNWwtNC45LTVMNS4yLDUuOXoiLz4KPHBhdGggY2xhc3M9InN0MCIgZD0iTTAtMC42aDE4djE4SDBWLTAuNnoiLz4KPC9zdmc+Cg'); background-position-x: 100%; background-position-y: 50%; background-repeat: no-repeat; background-size: 20px auto; box-shadow: none; box-sizing: border-box; border-radius: 0px 0px 0px 0px; height: 28px; width: 15em; line-height: 16px; font-size: 13px;"
        onchange="vor_select()"
        >
        <option value="41.770528 8.774667 8 ajo">Ajaccio VOR-DME</option>
        <option value="50.902222 4.538056 9 bub">Brussels VOR-DME</option>
        <option value="32.747039 -16.705686 9 fun">Madeira VOR-DME</option>
        <option value="-22.315389 166.473167 9 mga">Ouere VOR-DME</option>
        <option value="-31.673889 116.017222 9 pea">Pearce TACAN</option>
        <option value="15.00863889 -90.47033333 9 rab">Rabinal VOR-DME</option>
    </select>

    <div id="vormap" style="width: 100%; height: 500px; margin-bottom: 1em; margin-top: 1em"></div>

    <script>

        var vormap;
        var vormap_ready = false;
        var ajo, bub, fun, mga, pea, rab;

        function vor_select() {
            var inp = document.getElementById('vor_selector');
            var tab = inp.value.split(' ');
            vormap.setView([tab[0], tab[1]], tab[2]);
            eval(tab[3]).openPopup();
        };

        /* delay the creation of the map after the whole page has been loaded */
        setInterval(function() {

        if ((typeof (L) !== "undefined") & ~vormap_ready) {
        vormap = L.map( 'vormap', {
            center: [41.770528, 8.774667],
            scrollWheelZoom: false,
            zoom: 8 
        })

        L.tileLayer('http://stamen-tiles-a.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png', {
            maxZoom: 18,
            attribution:
            'Map tiles by <a href="http://stamen.com/">Stamen Design</a>, '+
            'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. '+
            'Data by <a href="http://openstreetmap.org/">OpenStreetMap</a>, '+
            'under <a href="http://creativecommons.org/licenses/by-sa/3.0">CC BY SA</a>.',
            id: 'stamen.terrain'
        }).addTo(vormap);


        L.geoJson(
            calibration_trajectories,
            style={
                color: '#FED976',
                weight: 2 
            }
        ).bindPopup(function (layer) {
            return "<b>Trajectory " 
              + layer.feature.properties.flight_id
              + "</b><br/> with aircraft "
              + layer.feature.properties.registration
              + " and callsign "
              + layer.feature.properties.callsign;
        }).addTo(vormap);

        L.geoJson(
            calibration_segments,
            {
               style: function (feature) {
                   return {
                       color: feature.properties.color,
                       weight: 2 
                  };
               }
            }
        ).addTo(vormap);

        ajo = L.marker([41.770528, 8.774667]);
        ajo.addTo(vormap);
        ajo.bindPopup('<b>Ajaccio VOR-DME</b>')
        bub = L.marker([50.902222, 4.538056]);
        bub.addTo(vormap);
        bub.bindPopup('<b>Brussels VOR-DME</b>')
        fun = L.marker([32.747039, -16.705686]);
        fun.addTo(vormap);
        fun.bindPopup('<b>Madeira VOR-DME</b>')
        mga = L.marker([-22.315389, 166.473167]);
        mga.addTo(vormap);
        mga.bindPopup('<b>Ouere VOR-DME</b>')
        pea = L.marker([-31.673889, 116.017222]);
        pea.addTo(vormap);
        pea.bindPopup('<b>Pearce TACAN</b>')
        rab = L.marker([15.00863889, -90.47033333]);
        rab.addTo(vormap);
        rab.bindPopup('<b>Rabinal VOR-DME</b>')

        ajo.openPopup();
        vormap_ready = true;

      }})

    </script>



.. [3] Time, distance and bearing thresholds may need further ajustments for a proper picture. Note the kiruna14 seems to circle around a position that is not referenced in the database. Any help or insight welcome!

Equipped aircraft for beacon calibration
========================================

This list only contains the equipped aircraft for the calibration in the sample
dataset. Apart from F-HNAV, registration numbers were found on social networks.
Two of the aircraft registrations were not in the provided database at the time
of the writing, so we added them manually.


.. raw:: html

   <blockquote class="twitter-tweet" style="width: 100%"><p lang="en" dir="ltr">Ever seen a Runway Calibration ? <a href="https://twitter.com/Beechcraft?ref_src=twsrc%5Etfw">@Beechcraft</a> <a href="https://twitter.com/hashtag/B300?src=hash&amp;ref_src=twsrc%5Etfw">#B300</a> VH-FIZ was out calibrating RWY16L <a href="https://twitter.com/SydneyAirport?ref_src=twsrc%5Etfw">@SydneyAirport</a> this morning, making a number of approaches to calibrate and certify ILS precision 💯. The orbits in the track are done to fit in with regular arrivals and departures ✈️💙📷 <a href="https://t.co/CRkXfsscHa">pic.twitter.com/CRkXfsscHa</a></p>&mdash; 16Right Media (@www16Right) <a href="https://twitter.com/www16Right/status/982854657151197185?ref_src=twsrc%5Etfw">April 8, 2018</a></blockquote>
   <!--<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>-->


.. code:: python

    from traffic.data.samples.calibration import traffic as calibration
    from traffic.data import aircraft

    # aircraft not in junzis database
    other_aircraft = {"4076f1": "G-TACN (DA62)", "750093": "9M-FCL (LJ60)"}

    (
        calibration.groupby(["flight_id"], as_index=False)
        .agg({"timestamp": "min", "icao24": "first"})
        .assign(
            registration=lambda df: df.icao24.apply(
                lambda x: f"{aircraft[x].regid.item()} ({aircraft[x].mdl.item()})"
                if aircraft[x].shape[0] > 0
                else other_aircraft.get(x, None)
            ),
            flight_id=lambda df: df.agg(
                # not the most efficient way but quite readable
                lambda x: f"{x.flight_id} ({x.timestamp:%Y-%M-%d})", axis=1
            )
        )
        .sort_values(["registration", "timestamp"])
        .groupby(["registration", "icao24"])
        .apply(lambda df: ", ".join(df.flight_id))
        .pipe(lambda series: pd.DataFrame({"flights": series}))
    )


.. raw:: html

   <table class="dataframe" border="0">
     <thead>
       <tr style="text-align: left;">
         <th></th>
         <th></th>
         <th>flights</th>
       </tr>
       <tr>
         <th>registration</th>
         <th>icao24</th>
         <th></th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <th>9M-FCL (LJ60)</th>
         <th>750093</th>
         <td>kota_kinabalu (2017-03-08)</td>
       </tr>
       <tr>
         <th>C-GFIO (CRJ2)</th>
         <th>c052bb</th>
         <td>vancouver (2018-10-06)</td>
       </tr>
       <tr>
         <th>C-GNVC (CRJ2)</th>
         <th>c06921</th>
         <td>montreal (2018-12-11)</td>
       </tr>
       <tr>
         <th>D-CFMD (B350)</th>
         <th>3cce6f</th>
         <td>munich (2019-03-04), vienna (2018-11-20)</td>
       </tr>
       <tr>
         <th>F-HNAV (BE20)</th>
         <th>39b415</th>
         <td>ajaccio (2018-01-12), monastir (2018-11-21), toulouse (2017-06-16), ...</td>
       </tr>
       <tr>
         <th>G-GBAS (DA62)</th>
         <th>4070f4</th>
         <td>london_heathrow (2018-01-12), lisbon (2018-11-13), funchal (2018-11-23)</td>
       </tr>
       <tr>
         <th>G-TACN (DA62)</th>
         <th>4076f1</th>
         <td>cardiff (2019-02-15), london_gatwick (2019-02-28)</td>
       </tr>
       <tr>
         <th>SE-LKY (BE20)</th>
         <th>4ab179</th>
         <td>bornholm (2018-11-26), kiruna (2019-01-30)</td>
       </tr>
       <tr>
         <th>VH-FIZ (B350)</th>
         <th>7c1a89</th>
         <td>noumea (2017-11-05), perth (2019-01-22)</td>
       </tr>
       <tr>
         <th>YS-111-N (BE20)</th>
         <th>0b206f</th>
         <td>guatemala (2018-03-26), kingston (2018-06-26)</td>
       </tr>
     </tbody>
   </table>


.. raw:: html

   <div>
   <a class="reference internal image-reference" href="https://cdn.jetphotos.com/full/6/55737_1538774410.jpg" target='_blank'><img alt="F-HNAV" class="align-center" src="https://cdn.jetphotos.com/full/6/55533_1526410235.jpg" title="© Kris Van Craenenbroeck | Jetphotos" style="max-height: 200px; float: left; padding: 20px"></a>
   <a class="reference internal image-reference" href="https://www.jetphotos.com/photo/9094827" target='_blank'><img alt="C-GFIO" class="align-center" src="https://cdn.jetphotos.com/full/6/55737_1538774410.jpg" title="© Keeper1 | Jetphotos" style="max-height: 200px; display: block; padding: 20px"></a>
   </div>

