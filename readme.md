# A toolbox for manipulating and analysing air traffic data 

[![Documentation Status](https://readthedocs.org/projects/airtraffic/badge/?version=latest)](https://airtraffic.readthedocs.io/en/latest/?badge=latest)


The traffic library helps working with common sources of air traffic data.

Its main purpose is to offer basic cumbersome data analysis methods commonly
applied to trajectories and ATC sectors. When a specific function is not
provided, the access to the underlying structure is direct, through an attribute
pointing to a pandas dataframe.

The library also offers facilities to parse and/or access traffic data from open
sources of ADS-B traffic like the `OpenSky Network <https://opensky-network.org/>`_
or Eurocontrol DDR files. It is designed to be easily extendable to other
sources of data.

Eventually, static and dynamic output are available for Matplotlib, Google
Earth (kmz) and CesiumJS (czml).

## Installation

```
python setup.py install
```

## Documentation

Documentation available at [https://airtraffic.readthedocs.io/](https://airtraffic.readthedocs.io/)

