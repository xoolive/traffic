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

Eventually, static and dynamic output are available for Matplotlib, Google
Earth (kmz) and CesiumJS (czml).

## Installation

Latest release:

```
pip install traffic
```

Development version:

```
pip install git+https://github.com/xoolive/traffic
```

## Command line tool

The `traffic` tool scripts around the library for common usecases.

You may download data from OpenSky Impala shell (add the verbosity `-v` flag to
get messages if the connection fails or stalls):

```
traffic opensky 2018-01-01T06:00 -s 2018-01-01T08:00 -b Andorra -o andorra.pkl
```

You may quickly inspect the contents of the file:

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

## Documentation

Documentation available [here](https://airtraffic.readthedocs.io/)
