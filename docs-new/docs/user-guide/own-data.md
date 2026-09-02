# Loading your own data

`Flight` and `Traffic` wrap a pandas DataFrame without validating, renaming, or
converting anything. The DataFrame you pass in must already use the column names
that `traffic` expects.

## Required columns

The minimum set of columns is:

- `icao24`: the ICAO transponder ID of an aircraft;
- `callsign`: an identifier associated with the registration of an aircraft, its
  mission, or a route;
- `timestamp`: timezone-aware timestamps are preferable;
- `latitude`, `longitude`: degrees, WGS84 (EPSG:4326);
- `altitude`: feet. Data recorded in metres must be converted before use, for
  example, `altitude_ft = altitude_m / 0.3048`.

A `flight_id` column may be used in place of the (`icao24`, `callsign`) pair.
This is useful when a dataset has no `callsign` column.

## Loading a CSV file

Consider a CSV file named `example_flight_or_collection.csv` with columns
`icao24`, `datetime`, `lat`, `lon`, and `alt`. The columns must be renamed, and
the timestamp must be converted to a timezone-aware datetime:

```python
import pandas as pd
from traffic.core import Flight, Traffic

df = (
    pd.read_csv("example_flight_or_collection.csv")
    .rename(
        columns={
            "datetime": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "alt": "altitude",
        }
    )
    .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"], utc=True))
)

# Use Flight when the CSV contains one trajectory.
flight = Flight(df)

# Use Traffic when the CSV contains several trajectories.
traffic = Traffic(df)
```

Timezone-naive timestamps may work with some methods, but the behaviour is not
guaranteed. Converting to UTC with `utc=True` avoids this issue.
