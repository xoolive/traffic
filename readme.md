# `traffic` library for working with various sources of ATM data

The `traffic` library helps working with various sources of ATM data

- `traffic.algorithms` provides basic tools for working with trajectories (Douglas-Peucker, etc.)
- `traffic.core` provides basic structures for flights, set of flights (traffic) and sectors;
- `traffic.data` gives access to many sources of data. It includes basic information about European FIRs, aircraft (from [junzis](https://junzisun.com/adb/)), airports, airways and navaids. It also provides an interface to data from the [OpenSky Network](https://opensky-network.org/) and Eurocontrol DDR files.

The repository contains sample data. The sample `so6` file is built from ADS-B data (DF17 only, no MLAT) from the [OpenSky Network](https://opensky-network.org/), filtered through a Douglas-Peucker algorithm (tolerance of 1km) and given for illustration purposes only.

## Documentation

coming soon

In the meantime, you can check the `scripts` and `notebooks` directories.
