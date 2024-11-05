![A toolbox for processing and analysing air traffic data](./docs/_static/logo/logo_full.png)

[![Documentation Status](https://github.com/xoolive/traffic/workflows/docs/badge.svg)](https://traffic-viz.github.io/)
[![tests](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml/badge.svg?branch=master&event=push)](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)
![License](https://img.shields.io/pypi/l/traffic.svg)  
[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![PyPI version](https://img.shields.io/pypi/v/traffic)
[![PyPI downloads](https://img.shields.io/pypi/dm/traffic)](https://pypi.org/project/traffic)
![Conda version](https://img.shields.io/conda/vn/conda-forge/traffic)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/traffic.svg)](https://anaconda.org/conda-forge/traffic)  
[![JOSS paper](http://joss.theoj.org/papers/10.21105/joss.01518/status.svg)](https://doi.org/10.21105/joss.01518)

The traffic library helps to work with common sources of air traffic data.

Its main purpose is to provide data analysis methods commonly applied to trajectories and airspaces. When a specific function is not provided, the access to the underlying structure is direct, through an attribute pointing to a pandas dataframe.

The library also offers facilities to parse and/or access traffic data from open sources of ADS-B traffic like the [OpenSky Network](https://opensky-network.org/) or Eurocontrol DDR files. It is designed to be easily extendable to other sources of data.

Static visualization (images) exports are accessible via Matplotlib/Cartopy. More dynamic visualization frameworks are easily accessible in Jupyter environments with [ipyleaflet](http://ipyleaflet.readthedocs.io/) and [altair](http://altair-viz.github.io/); or through exports to other formats, including CesiumJS or Google Earth.

## Installation

**Full installation instructions are to be found in the [documentation](https://traffic-viz.github.io/installation.html).**

- If you are not familiar/comfortable with your Python environment, please install the latest `traffic` release in a new, fresh conda environment.

  ```sh
  conda create -n traffic -c conda-forge python=3.12 traffic
  ```

- Adjust the Python version you need (>=3.10) and append packages you need for working efficiently, such as Jupyter Lab, xarray, PyTorch or more.
- Then activate the environment every time you need to use the `traffic` library:

  ```sh
  conda activate traffic
  ```

**Warning!** Dependency resolution may be tricky, esp. if you use an old conda environment where you overwrote `conda` libraries with `pip` installs. **Please only report installation issues in new, fresh conda environments.**

If conda fails to resolve an environment in a reasonable time, consider using a [Docker image](https://traffic-viz.github.io/user_guide/docker.html) with a working installation.

For troubleshooting, refer to the appropriate [documentation section](https://traffic-viz.github.io/troubleshooting/installation.html).

## Credits

[![JOSS
badge](http://joss.theoj.org/papers/10.21105/joss.01518/status.svg)](https://doi.org/10.21105/joss.01518)

- Like [other researchers before](https://scholar.google.com/scholar?cites=18420568209924139259&scisbd=1), if you find this project useful for your research and use it in an academic work, you may cite it as:

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

- Additionally, you may consider adding a star to the repository. This token of appreciation is often interpreted as positive feedback and improves the visibility of the library.

## Documentation

[![Documentation Status](https://github.com/xoolive/traffic/workflows/docs/badge.svg)](https://traffic-viz.github.io/)
[![Join the chat at https://gitter.im/xoolive/traffic](https://badges.gitter.im/xoolive/traffic.svg)](https://gitter.im/xoolive/traffic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Documentation available at <https://traffic-viz.github.io/>  
Join the Gitter chat for assistance: https://gitter.im/xoolive/traffic

## Tests and code quality

[![tests](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml/badge.svg?branch=master&event=push)](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml)
[![Code Coverage](https://img.shields.io/codecov/c/github/xoolive/traffic.svg)](https://codecov.io/gh/xoolive/traffic)
[![Codacy Badge](https://img.shields.io/codacy/grade/eea673ed15304f1b93490726295d6de0)](https://www.codacy.com/manual/xoolive/traffic)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy.readthedocs.io/)

Unit and non-regression tests are written in the `tests/` directory. You may run `pytest` from the root directory.

Tests are checked on [Github Actions](https://github.com/xoolive/traffic/actions/workflows/run-tests.yml) platform upon each commit. Latest status and coverage are displayed with standard badges hereabove.

In addition to unit tests, code is checked against:

- linting and formatting with [ruff](https://beta.ruff.rs/docs/);
- static typing with [mypy](https://mypy.readthedocs.io/)

[pre-commit](https://pre-commit.com/) hooks are available in the repository.

## Feedback and contribution

Any input, feedback, bug report or contribution is welcome.

- Should you encounter any [issue](https://github.com/xoolive/traffic/issues/new), you may want to file it in the [issue](https://github.com/xoolive/traffic/issues/new) section of this repository.
- If you intend to [contribute to traffic](https://traffic-viz.github.io/installation.html#contribute-to-traffic) or file a pull request, the best way to ensure continuous integration does not break is to reproduce an environment with the same exact versions of all dependency libraries. Please follow the [appropriate section](https://traffic-viz.github.io/installation.html#contribute-to-traffic) in the documentation.

  Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions.
