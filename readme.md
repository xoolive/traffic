# A toolbox for manipulating and analysing air traffic data

[![Documentation Status](https://readthedocs.org/projects/airtraffic/badge/?version=latest)](https://airtraffic.readthedocs.io/en/latest/?badge=latest)

The traffic library helps working with common sources of air traffic data.

Its main purpose is to offer basic cumbersome data analysis methods commonly
applied to trajectories and ATC sectors. When a specific function is not
provided, the access to the underlying structure is direct, through an attribute
pointing to a pandas dataframe.

The library also offers facilities to parse and/or access traffic data from open
sources of ADS-B traffic like the [OpenSky Network](https://opensky-network.org/)
or Eurocontrol DDR files. It is designed to be easily extendable to other
sources of data.

Static visualisation (images) exports are accessible via Matplotlib/Cartopy.
More dynamic visualisation frameworks are easily accessible in Jupyter
environments with [ipyleaflet](http://ipyleaflet.readthedocs.io/) and
[altair](http://altair-viz.github.io/); or through exports to other formats,
including CesiumJS or Google Earth.

## Installation

Latest release:

```
pip install traffic
```

Development version:

```
pip install git+https://github.com/xoolive/traffic
```

**Warning:** `cartotools` and `shapely` have strong dependencies to dynamic
libraries which may not be available on your system by default.

Before reporting an issue, please try to use an Anaconda environment. Other
installations may work but the Anaconda way proved to cause much less issues.

```
conda install cartopy shapely
```

## Command line tool

The `traffic` tool scripts around the library for common usecases.

You may download data from OpenSky Impala shell (check the help with option
`-h`):

```
traffic opensky 2018-01-01T06:00 -s 2018-01-01T08:00 -b Andorra -o andorra.pkl
```

You may inspect the contents of the file:

```
$ traffic show andorra.pkl
Traffic with 7 identifiers
                 count
icao24 callsign
3443d1 VLG242N     123
392ae7 AFR75SD     123
3444d2 VLG85VY     122
344691 VLG8988     117
4ca574 IBK5358     109
4ca7f2 IBK5FM      101
3443d2 VLG7964      85
```

More details in the [documentation](https://airtraffic.readthedocs.io/).

## Graphical user interface

A Qt application is provided for exploring and recording data.  
More details in the [GUI section of the documentation](https://airtraffic.readthedocs.io/en/latest/gui.html).

![GUI screenshot](https://raw.githubusercontent.com/xoolive/traffic/master/docs/_static/gui_start.png)

## Documentation

Documentation available [here](https://airtraffic.readthedocs.io/)

## Frequently asked questions

- Something doesn't work. What should I do?

Please file an [issue](https://github.com/xoolive/traffic/issues/new) but
activating the `DEBUG` messages first may be helpful:

```python
from traffic.core.logging import loglevel
loglevel('DEBUG')
```

- I encountered this issue, here is how to fix it.

First of all, thank you. All kinds of corrections are welcome.

I can include your fix in the code and push it. But since you did the job, you
may want to file a [PR](https://yangsu.github.io/pull-request-tutorial/) and
keep the authorship. It is also easier for me to keep track of corrections and
remember why things are the way they are.

- I want to know more about Eurocontrol NM files

 We download those files from Eurocontrol [Network Manager DDR2 repository
 service](https://www.eurocontrol.int/articles/ddr2-web-portal) under Dataset
 Files > Airspace Environment Datasets. You may not be entitled access to those
 data.

Should you have no such access, basic FIRs are provided in `eurofirs` from
`traffic.data`.

- I want to know more about Eurocontrol AIXM files

When you import `aixm_airspaces` from `traffic.data`, you need to set a path to
a directory containing AIRAC files. These are XML files following the
[AIXM](http://aixm.aero/) standard and produced by Eurocontrol. We download
those files from  Eurocontrol [Network Manager B2B web
service](https://eurocontrol.int/service/network-manager-business-business-b2b-web-services).
You may not be entitled access to those data.
