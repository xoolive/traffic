# A toolbox for processing and analysing air traffic data

[![Documentation Status](https://github.com/xoolive/traffic/workflows/docs/badge.svg)](https://traffic-viz.github.io/)
[![Build Status](https://travis-ci.org/xoolive/traffic.svg?branch=master)](https://travis-ci.org/xoolive/traffic)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic)
[![Codacy Badge](https://img.shields.io/codacy/grade/eea673ed15304f1b93490726295d6de0)](https://www.codacy.com/manual/xoolive/traffic)\
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![License](https://img.shields.io/pypi/l/traffic.svg)
[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)\
[![JOSS paper](http://joss.theoj.org/papers/10.21105/joss.01518/status.svg)](https://doi.org/10.21105/joss.01518)
![PyPI version](https://img.shields.io/pypi/v/traffic)
[![PyPI downloads](https://img.shields.io/pypi/dm/traffic)](https://pypi.org/project/traffic)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/traffic-viz/traffic_static/blob/master/notebooks/quickstart.ipynb)

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

```sh
pip install --upgrade traffic
```

Development version:

```sh
pip install git+https://github.com/xoolive/traffic
```

**Warning:** `cartes` and `shapely` have strong dependencies to dynamic
libraries which may not be available on your system by default.

Before reporting an issue, please try to use an Anaconda environment. Other
installations (You may check them in the `.travis.yml` configuration file.)
should work but the Anaconda way proved to work smoothly.

```sh
conda install cartopy shapely
```

For troubleshootings, refer to the appropriate
[documentation section](https://traffic-viz.github.io/installation.html#troubleshooting).

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

Additionally, you may consider adding a star to the repository. This token of appreciation is often interpreted as a positive feedback and improves the visibility of the library.

## Documentation

[![Documentation Status](https://github.com/xoolive/traffic/workflows/docs/badge.svg)](https://traffic-viz.github.io/)
[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Documentation available at [https://traffic-viz.github.io/](https://traffic-viz.github.io/)\
Join the Gitter chat: https://gitter.im/xoolive/traffic

## Tests and code quality

[![Build Status](https://travis-ci.org/xoolive/traffic.svg?branch=master)](https://travis-ci.org/xoolive/traffic)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic)
[![Codacy Badge](https://img.shields.io/codacy/grade/eea673ed15304f1b93490726295d6de0)](https://www.codacy.com/manual/xoolive/traffic)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)

Unit and non-regression tests are written in the `tests/` directory. You may run
`pytest` or `tox` from the root directory. Tests are currently performed with
Python 3.6 and 3.7.

Tests are checked on [travis continuous integration](https://travis-ci.org/xoolive/traffic)
platform upon each commit. Latest status and coverage are displayed with standard
badges hereabove.

In addition, code is checked against static typing with [mypy](https://mypy.readthedocs.io/)
([pre-commit](https://pre-commit.com/) hooks are available in the repository) and
extra quality checks performed by [Codacy](https://www.codacy.com/manual/xoolive/traffic).

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
from traffic.core import loglevel
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

[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

- I want to know more about Eurocontrol NM files

We download these files from Eurocontrol [Network Manager Demand Data
Repository (DDR)](https://www.eurocontrol.int/ddr) under Dataset Files >
Airspace Environment Datasets. [Access
conditions](https://www.eurocontrol.int/ddr#access-conditions) are managed by
EUROCONTROL.

Should you have no such access, basic FIRs are provided in `eurofirs` from
`traffic.data`.

- I want to know more about Eurocontrol AIXM files

When you import `aixm_airspaces` from `traffic.data`, you need to set a path
to a directory containing AIRAC files. These are XML files following the
[AIXM](http://aixm.aero/) standard and produced by Eurocontrol. We download
these files from Eurocontrol [Network Manager B2B web
services](https://eurocontrol.int/service/network-manager-business-business-b2b-web-services).
You have to own a B2B certificate granted by EUROCONTROL to get access to
this data.

- What does AIRAC mean?

Aeronautical Information Publications are updated every 28 days according to
fixed calendar. This cycle is known as AIRAC (Aeronautical Information
Regulation And Control) cycle.
