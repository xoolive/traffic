# A toolbox for processing and analysing air traffic data

[![Documentation Status](https://readthedocs.org/projects/traffic-viz/badge/?version=latest)](https://traffic-viz.github.io/)
[![Build Status](https://travis-ci.org/xoolive/traffic.svg?branch=master)](https://travis-ci.org/xoolive/traffic)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic) 
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)
![License](https://img.shields.io/pypi/l/traffic.svg)\
[![JOSS badge](http://joss.theoj.org/papers/10.21105/joss.01518/status.svg)](https://doi.org/10.21105/joss.01518)


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

## Credits

[![JOSS badge](http://joss.theoj.org/papers/10.21105/joss.01518/status.svg)](https://doi.org/10.21105/joss.01518)

If you find this project useful for your research and use it in an academic
work, you may cite it as:

```bibtex
@article{olive2019traffic,
    author={Xavier {Olive}},
    journal={Journal of Open Source Software},
    title={traffic, a toolbox for processing and analysing air traffic data},
    year={2019},
    volume={4},
    pages={1518},
    doi={10.21105/joss.01518},
    issn={2475-9066},
}
```

## Documentation

[![Documentation Status](https://readthedocs.org/projects/traffic-viz/badge/?version=latest)](https://traffic-viz.github.io/)

Documentation available at [https://travis-viz.github.io/](https://traffic-viz.github.io/)

## Running tests

[![Build Status](https://travis-ci.org/xoolive/traffic.svg?branch=master)](https://travis-ci.org/xoolive/traffic)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic) 

Unit and non-regression tests are written in the `tests/` directory. You may run
`pytest` or `tox` from the root directory. Tests are currently performed with 
Python 3.6 and 3.7.

Tests are checked on [travis continuous integration](https://travis-ci.com/)
platform upon each commit. Latest status and coverage are displayed with standard
badges hereabove.

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

## Feedback and contribution

Any input, feedback, bug report or contribution is welcome.

Should you encounter any issue, you may want to file it in the [issue](https://github.com/xoolive/traffic/issues/new) section of this repository. Please first activate the `DEBUG` messages recorded using Python logging mechanism with the following snippet:

```python
from traffic.core.logging import loglevel
loglevel('DEBUG')
```

Bug fixes and improvements in the library are also helpful.

If you share a fix together with the issue, I can include it in the code for
you. But since you did the job, pull requests (PR) let you keep the authorship
on your additions. For details on creating a PR see GitHub documentation
[Creating a pull
request](https://help.github.com/en/articles/creating-a-pull-request). You can
add more details about your example in the PR such as motivation for the example
or why you thought it would be a good addition. You will get feed back in the PR
discussion if anything needs to be changed. To make changes continue to push
commits made in your local example branch to origin and they will be
automatically shown in the PR.

You may find the process troublesome but please keep in mind it is actually
easier that way to keep track of corrections and to remember why things are the
way they are.

## Frequently asked questions

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
