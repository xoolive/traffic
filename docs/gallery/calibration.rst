.. raw:: html

  <div style="display: none">

.. jupyter-execute::

  # This trick only helps to trigger the proper Javascript library require queries.
  from ipyleaflet import Map
  Map()

.. raw:: html

  </div>

.. raw:: html

    <script src="../_static/calibration.geojson"></script>

Calibration flights
-------------------

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

.. jupyter-execute::

   from traffic.data.samples.calibration import ajaccio
   from traffic.data import navaids

   navaids.extent(ajaccio).query('type == "VOR"')

Next step is to compute for each point the distance and bearing from the VOR to
each point of the trajectory. The parts of the trajectory that are of interest
are the ones with little to no variation in the distance (circles) and in the
bearing (radials) to the VOR.

Then, we can write a simple .query() followed by a .split() method to select all
segments with a constant bearing and distance with respect to the selected VOR.

.. jupyter-execute::

    from functools import reduce
    from operator import or_

    vor = navaids.extent(ajaccio)['AJO']

    ajaccio = (
        ajaccio.distance(vor)  # add a distance column (in nm) w.r.t the VOR
        .bearing(vor)  # add a bearing column w.r.t the VOR
        .assign(
            distance_diff=lambda df: df.distance.diff().abs(),  # large circles
            bearing_diff=lambda df: df.bearing.diff().abs(),  # long radials
        )
    )

    constant_distance = list(
        segment for segment in ajaccio.query('distance_diff < .02').split('1 min')
        if segment.longer_than('5 minutes')
    )

    # trick to display many trajectories
    reduce(or_, constant_distance)


We have all we need to enhance the interesting parts of the trajectory now:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    from cartes.crs import Lambert93, EuroPP
    from cartes.utils.features import countries

    from traffic.data import airports

    point_params = dict(zorder=5, text_kw=dict(fontname="Ubuntu", fontsize=15))
    box_params = dict(boxstyle="round", facecolor="lightpink", alpha=0.7, zorder=5)

    with plt.style.context("traffic"):

        fig, ax = plt.subplots(subplot_kw=dict(projection=EuroPP()))

        ax.add_feature(countries())

        # airport information
        airports["LFKJ"].point.plot(ax, **point_params)

        # VOR information
        shift_vor = dict(units="dots", x=20, y=10)
        vor.plot(ax, marker="h", shift=shift_vor, **point_params)

        # full trajectory in dashed lines
        ajaccio.plot(ax, color="#aaaaaa", linestyle="--")

        # constant distance segments
        for segment in ajaccio.query("distance_diff < .02").split("1 minute"):
            if segment.longer_than("3 minutes"):
                segment.plot(ax, color="crimson")

                # an annotation with the radius of the circle
                distance_vor = segment.data.distance.mean()
                segment.at().plot(
                    ax,
                    alpha=0,  # We don't need the point, only the text
                    text_kw=dict(s=f"{distance_vor:.1f} nm", bbox=box_params),
                )

        # constant bearing segments
        for segment in ajaccio.query("bearing_diff < .01").split("1 minute"):
            if segment.longer_than("3 minutes"):
                segment.plot(ax, color="forestgreen")

        ax.set_extent((7.6, 9.9, 41.2, 43.3))
        ax.spines["geo"].set_visible(False)


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


.. jupyter-execute::

    from traffic.core import Traffic
    from traffic.data import aircraft
    from traffic.data.samples import calibration

    (
        Traffic.from_flights(
            getattr(calibration, name).assign(
                # create a flight_id which includes the date of the flight
                flight_id=lambda df: f"{name} ({df.timestamp.min():%Y-%m-%d})"
            )
            for name in calibration.__all__
        )
        .summary(["flight_id", "icao24", "start"])
        .eval()
        .merge(aircraft.data)
        .sort_values(["registration", "start"])
        .groupby(["registration", "typecode", "icao24"])
        .apply(lambda df: ", ".join(df.flight_id))
        .to_frame()
    )

.. raw:: html

   <div>
   <a class="reference internal image-reference" href="https://cdn.jetphotos.com/full/6/55737_1538774410.jpg" target='_blank'><img alt="F-HNAV" class="align-center" src="https://cdn.jetphotos.com/full/6/55533_1526410235.jpg" title="© Kris Van Craenenbroeck | Jetphotos" style="max-height: 200px; float: left; padding: 20px"></a>
   <a class="reference internal image-reference" href="https://www.jetphotos.com/photo/9094827" target='_blank'><img alt="C-GFIO" class="align-center" src="https://cdn.jetphotos.com/full/6/55737_1538774410.jpg" title="© Keeper1 | Jetphotos" style="max-height: 200px; display: block; padding: 20px"></a>
   </div>
