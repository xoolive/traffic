# A toolbox for processing and analysing air traffic data

[![Documentation Status](https://readthedocs.org/projects/traffic-viz/badge/?version=latest)](https://traffic-viz.github.io/)
[![Build Status](https://travis-ci.org/xoolive/traffic.svg?branch=master)](https://travis-ci.org/xoolive/traffic)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic) 
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)


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
installations (You may check them in the `.travis.yml` configuration file.)
should work but the Anaconda way proved to work smoothly.

```
conda install cartopy shapely
```

## Documentation

Documentation available [here](https://traffic-viz.github.io/)

## Command line tool

The `traffic` tool scripts around the library for common usecases.

The most basic use case revolves around exploring the embedded data. You may check
the help with `traffic data -h`.

```
traffic data -p Tokyo
     altitude country iata  icao   latitude   longitude                                name
3820       21   Japan  HND  RJTT  35.552250  139.779602  Tokyo Haneda International Airport
3821      135   Japan  NRT  RJAA  35.764721  140.386307  Tokyo Narita International Airport
```

More details in the [documentation](https://traffic-viz.github.io/).

## Graphical user interface

A Qt application is provided for exploring and recording data.  
More details in the [GUI section of the documentation](https://traffic-viz.github.io/gui.html).

![GUI screenshot](https://raw.githubusercontent.com/xoolive/traffic/master/docs/_static/gui_start.png)

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
