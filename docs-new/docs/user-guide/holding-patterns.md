# How to detect holding patterns in aircraft trajectories?

Holding patterns are racetrack-shaped flight paths that an aircraft follows while awaiting further instructions or clearance from air traffic control (ATC). They are typically used to delay an aircraft’s approach or to maintain flight without progressing towards its destination, often due to airport congestion, adverse weather conditions, or other operational factors.

The `traffic` library provides a function to detect holding patterns in trajectories: [`holding_pattern()`](https://traffic-viz.github.io/api_reference/traffic.core.flight.html#traffic.core.Flight.holding_pattern). The function returns a `FlightIterator` as there may be several go-arounds in one flight.

To give an example of the function, let's start by using a sample trajectory: [`belevingsvlucht`](https://traffic-viz.github.io/data_sources/samples.html#belevingsvlucht).

```python
from traffic.core.flight import Flight
from traffic.core.iterator import FlightIterator
from traffic.data.samples import belevingsvlucht

hp: FlightIterator = belevingsvlucht.holding_pattern()
hp
```

To get only the first holding pattern from the `FlightIterator`, you can do it in three ways:

```python
hp: FlightIterator = belevingsvlucht.holding_pattern()
hp.next()
```

or

```python
hp: Flight | None = belevingsvlucht.holding_pattern().next()
hp
```

or

```python
hp: Flight | None = belevingsvlucht.next("holding_pattern")
hp
```

Next, the `label()` method might be useful. This method adds a column _holding_ which is `True` when the trajectory follows a holding pattern.

```python
flight_with_label = belevingsvlucht.label("holding_pattern", holding_pattern=True)
flight_with_label.data.head()
```

Next, we can also visualize the time when the holding pattern happens in a flight using `altair`

```python
import altair as alt

alt.data_transformers.disable_max_rows()

base = (
    same_flight.chart()
    .encode(
        alt.X(
            "utchoursminutesseconds(timestamp)",
            axis=alt.Axis(format="%H:%M", title=None),
        )
    )
)

alt.vconcat(
    base.encode(
        alt.Y("holding_pattern:N", title="Holding pattern"),
        alt.Color("holding_pattern:N", legend=None),
    ).mark_point(size=30)
).resolve_scale(color="independent").configure_legend(
    title=None
).configure_axis(
    titleAngle=0,
    titleY=-15,
    titleX=0,
    titleAnchor="start",
    titleFont="Lato",
    titleFontSize=14,
    labelFontSize=12,
)
```

When we have multiple flights, like using `landing_zurich_2019` dataset, it can be quicker to filter out flights that do not have a self intersecting trajectory. Below is an example of flights that don't have and have self intersecting.

```python
from traffic.data.datasets import landing_zurich_2019
landing_zurich_2019[0] | landing_zurich_2019[1]
```

Evaluating `landing_zurich_2019[0].shape.is_simple` will return `False`, while evaluting `landing_zurich_2019[1].shape.is_simple` will return `True`.

Now let's try on the dataset (make sure you have imported it from the previous code block).

```python
def self_intersecting(flight: Flight) -> bool:
    """
    Check if a flight trajectory is self-intersecting.
    """
    shape = flight.shape
    if shape is None:
        return False
    return not shape.is_simple

subset = (
    landing_zurich_2019
    .longer_than("30 min")
    .pipe(self_intersecting)
    .resample("1s")
    .has("holding_pattern")
    .eval(desc="processing")
)

subset
```

Finally, we can get samples of flights that have holding patterns.

```python
subset[0] | subset[1] | subset[2]
```
