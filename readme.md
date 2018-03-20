`traffic` library for working with open (and closed) ATM data

- `traffic.data` gets information about sectorisation of airspaces
  (AIRAC files needed), airports, airways and navaids.

- `traffic.so6` works with Eurocontrol DDR files.

- `traffic.tools` provides basic tools for working with trajectories.

The repository contains sample data. The sample `so6` file is built from ADS-B
data (DF17 only, no MLAT) from the [OpenSky Network](https://opensky-network.org/), filtered through a Douglas-Peucker algorithm (tolerance of 1km) and given for illustration purposes only.  
Columns of the so6 that are not parsed are assigned value `0`.
