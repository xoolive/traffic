# Declarative trajectory processing

Basic operations on `Flight` objects define a specific
language which enables to express programmatically any kind of preprocessing.
The downside with programmatic preprocessing is that it may become unnecessarily
complex because of safeguards, nested loops and conditions necessary to express
even basic treatments.

The main issue with the code above is that **code for preprocessing and code for
visualization** are strongly connected: it is impossible to produce a
visualization without running “heavy” processing, as subsets of trajectories are
never stored as `Traffic` collections for future reuse.

There are several ways to collect trajectories:

- with trajectory arithmetic: the `+` operator (and therefore the
  sum() Python built-in function) between `Flight` and
  `Traffic` objects always returns a new
  `Traffic` object;

- the `Traffic.from_flights()` class method builds a
  `Traffic` object from an iterable structure of
  `Flight` objects. It is more robust than the sum()
  Python function as it will ignore `None` objects which may be found in the
  iterable.

```python
from traffic.core import Traffic


def select_landing(airport):
    for flight in quickstart:
        if low_alt := flight.query("altitude < 3000"):
            if not pd.isna(v_mean := low_alt.vertical_rate_mean) and v_mean < -500:
                if low_alt.intersects(airport):
                    if low_alt.landing(airport).has():
                        yield low_alt.last("10 min")


Traffic.from_flights(select_landing(airports["LFPO"]))
```

!!! tip
Lazy iteration (`traffic.core.lazy`) offers flattened specifications of
trajectory preprocessing operations. Operations are stacked before being
evaluated in a single iteration, using multiprocessing if needed, only after
the specification is fully described.

    *Lazy evaluation* is a common wording in functional programming languages.
    It refers to a mechanism where the actual evaluation is deferred.

When you stack any `Flight` method returning `None | Flight` or a boolean, a
lazy iteration is triggered. You may
remember that:

- Most `Flight` methods returning a `Flight`, a boolean
  or `None` can be stacked on `Traffic` structures;
- When such a method is stacked, it is **not** evaluated, just pushed
  for later evaluation;
- The final `.eval()` call starts one single iteration and apply all
  stacked method to every `Flight` it can iterate on.
- If one of the methods returns `False` or `None`, the
  `Flight` is discarded;
- If one of the methods returns `True`, the `Flight` is
  passed as is to the next method.

The landing trajectory selection rewrites as:

```python
(
    quickstart.query("altitude < 3000")      # Traffic -> None | Traffic
    # Lazy iteration is triggered here by the .feature_lt method
    .feature_lt("vertical_rate_mean", -500)  # Flight -> None | Flight
    .intersects(airports["LFPO"])            # Flight -> bool
    .has('landing("LFPO")')                  # Flight -> bool
    .last("10 min")                          # Flight -> None | Flight
    # Now evaluation is triggered on 4 cores
    .eval(max_workers=4)  # the desc= argument creates a progress bar
)
```

!!! note
The `Flight.landing()` call (without considerations on vertical rate and
intersections) is already enough for many practical cases; more methods are
stacked here for explanatory purposes.

For reference, look at the subtle differences between the following processing:

- take the last 10 minutes of trajectories landing at LFPO (similar to above):

```python
t1 = (
    quickstart
    .has("landing('LFPO')")
    .last('10 min')
    .eval(max_workers=4)
)

with plt.style.context('traffic'):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    t1.plot(ax, color="#f58518")
    airports['LFPO'].plot(
        ax, footprint=False,
        runways=dict(linewidth=1, color='black', zorder=3)
    )
    ax.spines['geo'].set_visible(False)
```

- take the last minute of the segment of trajectory which is aligned on runway 06:

```python
t2 = (
    quickstart
    .next('landing("LFPO")')
    .query("ILS == '06'")
    .last("1 min")
    .eval(max_workers=4)
)

with plt.style.context('traffic'):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    t2.plot(ax, color="#f58518")
    airports['LFPO'].plot(ax, labels=dict(fontsize=11))
    ax.spines['geo'].set_visible(False)
```

- select full trajectories landing on runway 06 from one minute before landing:

```python
import pandas as pd


def last_minute_with_taxi(flight: "Flight") -> "None | Flight":
    for segment in flight.landing("LFPO"):
        if segment.ILS_max == "06":
            return flight.after(segment.stop - pd.Timedelta("1 min"))


t3 = quickstart.iterate_lazy().pipe(last_minute_with_taxi).eval()

with plt.style.context('traffic'):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    t3.plot(ax, color="#f58518", zorder=3)
    airports['LFPO'].plot(ax, labels=dict(fontsize=11))
    ax.spines['geo'].set_visible(False)
```

- select trajectories with more than one runway alignment at LFPG:

```python
def more_than_one_alignment(flight: "Flight") -> "None | Flight":
    segments = flight.landing("LFPG")
    if first := next(segments, None):
        if second := next(segments, None):
            return flight.after(first.start - pd.Timedelta('90s'))


t4 = quickstart.iterate_lazy().pipe(more_than_one_alignment).eval()

flight = t4[0]
segments = flight.landing("LFPG")
first = next(segments)
forward = first.first("70s").forward(minutes=4)

chart = (
    alt.layer(
        airports["LFPG"].geoencode(
            footprint=False,
            runways=dict(strokeWidth=1),
            labels=dict(fontSize=10),
        ),
        flight.geoencode().mark_line(stroke="#bab0ac"),
        forward.geoencode(stroke="#79706e", strokeDash=[7, 3], strokeWidth=0.8),
        first.geoencode().encode(alt.Color("ILS")),
        next(segments).geoencode().encode(alt.Color("ILS")),
    )
    .properties(
        title=f"Runway change at LFPG airport with {flight.callsign}",
        width=600,
    )
    .configure_view(stroke=None)
    .configure_legend(orient="bottom")
    .configure_title(font="Lato", fontSize=16, anchor="start")
)
chart
```

---

<nav class="tutorial-nav" aria-label="Tutorial navigation">
  <a class="prev-link" href="/tutorials/patterns/"><- Landing patterns</a>
</nav>
