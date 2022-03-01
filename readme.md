![A toolbox for processing and analysing air traffic data](./docs/_static/logo/logo_full.png)

[![Documentation Status](https://github.com/xoolive/traffic/workflows/docs/badge.svg)](https://traffic-viz.github.io/)
[![tests](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml/badge.svg?branch=master&event=push)](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![License](https://img.shields.io/pypi/l/traffic.svg)
[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)\
![PyPI version](https://img.shields.io/pypi/v/traffic)
[![PyPI downloads](https://img.shields.io/pypi/dm/traffic)](https://pypi.org/project/traffic)
![Conda version](https://img.shields.io/conda/vn/conda-forge/traffic)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/traffic.svg)](https://anaconda.org/conda-forge/traffic)\
[![JOSS paper](http://joss.theoj.org/papers/10.21105/joss.01518/status.svg)](https://doi.org/10.21105/joss.01518)

The traffic library helps working with common sources of air traffic data.

Its main purpose is to provide data analysis methods commonly applied to
trajectories and airspaces. When a specific function is not provided, the access
to the underlying structure is direct, through an attribute pointing to a pandas
dataframe.

The library also offers facilities to parse and/or access traffic data from open
sources of ADS-B traffic like the [OpenSky
Network](https://opensky-network.org/) or Eurocontrol DDR files. It is designed
to be easily extendable to other sources of data.

Static visualisation (images) exports are accessible via Matplotlib/Cartopy.
More dynamic visualisation frameworks are easily accessible in Jupyter
environments with [ipyleaflet](http://ipyleaflet.readthedocs.io/) and
[altair](http://altair-viz.github.io/); or through exports to other formats,
including CesiumJS or Google Earth.

## Installation

Full installation instructions are in the [documentation](https://traffic-viz.github.io/installation.html).

If you are not familiar/comfortable with your Python environment, please install `traffic` latest release in a new, fresh conda environment.

```sh
conda create -n traffic -c conda-forge python=3.9 traffic
```

Adjust the Python version you need (>=3.7) and append packages you need for working efficiently, such as Jupyter Lab, xarray, PyTorch or more.

Then activate the environment every time you need to use the `traffic` library:

```sh
conda activate traffic
```

**Warning!**

Dependency resolution may be tricky, esp. if you use an old conda environment
where you overwrote `conda` libraries with `pip` installs. **Please only report
installation issues in new, fresh conda environments.**

For troubleshooting, refer to the appropriate
[documentation section](https://traffic-viz.github.io/troubleshooting/installation.html).

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

Additionally, you may consider adding a star to the repository. This token of
appreciation is often interpreted as a positive feedback and improves the
visibility of the library.

## Documentation

[![Documentation Status](https://github.com/xoolive/traffic/workflows/docs/badge.svg)](https://traffic-viz.github.io/)
[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Documentation available at [https://traffic-viz.github.io/](https://traffic-viz.github.io/)\
Join the Gitter chat: https://gitter.im/xoolive/traffic

## Tests and code quality

[![tests](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml/badge.svg?branch=master&event=push)](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic)
[![Codacy Badge](https://img.shields.io/codacy/grade/eea673ed15304f1b93490726295d6de0)](https://www.codacy.com/manual/xoolive/traffic)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)

Unit and non-regression tests are written in the `tests/` directory. You may run
`pytest` from the root directory.

Tests are checked on [Github
Actions](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml)
platform upon each commit. Latest status and coverage are displayed with
standard badges hereabove.

In addition, code is checked against static typing with
[mypy](https://mypy.readthedocs.io/) ([pre-commit](https://pre-commit.com/)
hooks are available in the repository) and extra quality checks performed by
[Codacy](https://www.codacy.com/manual/xoolive/traffic).

## Feedback and contribution

Any input, feedback, bug report or contribution is welcome.

Should you encounter any issue, you may want to file it in the
[issue](https://github.com/xoolive/traffic/issues/new) section of this
repository. Please first activate the `DEBUG` messages recorded using Python
logging mechanism with the following snippet:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Bug fixes and improvements in the library are also always helpful.

If you share a fix together with the issue, I can include it in the code for
you. But since you did the job, pull requests (PR) let you keep the authorship
on your additions. For details on creating a PR see GitHub documentation
[Creating a pull
request](https://help.github.com/en/articles/creating-a-pull-request). You can
add more details about your example in the PR such as motivation for the example
or why you thought it would be a good addition. You will get feedback in the PR
discussion if anything needs to be changed. To make changes continue to push
commits made in your local example branch to origin and they will be
automatically shown in the PR.

You may find the process troublesome but please keep in mind it is actually
easier that way to keep track of corrections and to remember why things are the
way they are.
