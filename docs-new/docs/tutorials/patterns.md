# Low-altitude trajectory patterns in Paris metropolitan area

The `quickstart` dataset contains a collection of low-altitude trajectories.
In this section, the goal is to identify and visualize approach/departure
patterns in the Paris metropolitan area.

It is usually best to start by plotting the data as-is before adding
classification logic.

```python
with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))
    quickstart.plot(ax, alpha=0.7)
```

We see here several flows converging mostly in the two major airports in Paris
(i.e., Orly `LFPO` and Charles-de-Gaulle `LFPG`). However, more airports are
also visible, e.g. Beauvais airport to the North.

We can try to put a different colour to landing trajectories and take-off
trajectories to make this plot more meaningful. A first trick could be to pick a
colour based on the vertical rate average value.

A first heuristic is to color trajectories by mean vertical rate. It is not
perfect, but it gives a useful first separation between likely arrivals,
departures, and level segments.

```python
import pandas as pd

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

    for flight in quickstart:
        if pd.isna(flight.vertical_rate_mean):
            continue
        if flight.vertical_rate_mean < -500:
            flight.plot(ax, color="#4c78a8", alpha=0.5)
        elif flight.vertical_rate_mean > 1000:
            flight.plot(ax, color="#f58518", alpha=0.5)
        else:
            flight.plot(ax, color="#54a24b", alpha=0.5)
```

This approach is not perfect (there are quite some green trajectories) but gives
a good first idea of how traffic organizes itself. Let's try to focus on the
traffic to and from one airport, e.g. `LFPO`, in order to refine the
methodology.

A first approach to select those trajectories would be to pick the first/last
point of a `Flight` and check whether it falls within the
geographical scope of the airport. In the following snippet, we do things a bit
differently: we check whether the first/last 5 minutes of the trajectory
intersects the shape of the airport.

```python
from traffic.data import airports

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=Lambert93()))

    for flight in quickstart:
        if pd.isna(flight.vertical_rate_mean):
            continue
        if flight.vertical_rate_mean < -500:
            if flight.last("5 min").intersects(airports["LFPO"]):
                flight.plot(ax, color="#4c78a8", alpha=0.5)
        elif flight.vertical_rate_mean > 1000:
            if flight.first("5 min").intersects(airports["LFPO"]):
                flight.plot(ax, color="#f58518", alpha=0.5)
```

What is now becoming confusing is that there seems to have been a change in
runway configuration during the time interval covered by the dataset. It would
now probably become more comfortable if we could identify the runway used by
aircraft for take off or landing.

`traffic` provides `Flight.landing()` for landing and `Flight.takeoff()` for
take-off. Both methods return a `FlightIterator`. If we assume each trajectory
has only one landing attempt in this interval, we can apply `.next()` to get
the first matching segment and extract runway information:

```python
from tqdm.rich import tqdm

information = []

for flight in tqdm(quickstart):
    if landing := flight.landing("LFPO").next():
        information.append(
            {
                "callsign": flight.callsign,
                "icao24": flight.icao24,
                "airport": "LFPO",
                "stop": landing.stop,
                "ILS": landing.ILS_max,
            }
        )
    elif landing := flight.landing("LFPG").next():
        information.append(
            {
                "callsign": flight.callsign,
                "icao24": flight.icao24,
                "airport": "LFPG",
                "stop": landing.stop,
                "ILS": landing.ILS_max,
            }
        )

stats = pd.DataFrame.from_records(information)
stats
```

It appears that there was a coordinated runway configuration change around
13:20Z in Paris airports. This suggests plotting how traffic organizes in both
configurations.

```python
with plt.style.context("traffic"):
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=Lambert93()))

    for flight in quickstart:
        if segment := flight.landing("LFPO").next():
            index = int(flight.stop <= pd.Timestamp("2021-10-07 13:30Z"))
            flight.plot(ax[index], color="#4c78a8", alpha=0.5)
        elif segment := flight.takeoff("LFPO").next():
            index = int(segment.start <= pd.Timestamp("2021-10-07 13:20Z"))
            flight.plot(ax[index], color="#f58518", alpha=0.5)
```

So it is now time to do a preliminary visualization with a basic background,
including administrative boundaries of Greater Paris Area and the Seine river as
an additional landmark:

```python
from cartes.atlas import france
from cartes.crs import Lambert93, PlateCarree
from cartes.osm import Nominatim


# background elements
paris_area = france.data.query("ID_1 == 1000")
seine_river = Nominatim.search("Seine river, France").shape.intersection(
    paris_area.union_all().buffer(0.1)
)

with plt.style.context("traffic"):
    fig, ax = plt.subplots(3, 2, figsize=(10, 15), subplot_kw=dict(projection=Lambert93()))

    airport_codes = ["LFPO", "LFPG", "LFPB"]
    for flight in quickstart:
        phases = flight.phases()

        if phases.query('phase == "DESCENT"'):
            for airport_index, airport in enumerate(airport_codes):
                if segment := flight.landing(airport).next():
                    time_index = int(segment.stop <= pd.Timestamp("2021-10-07 13:20Z"))
                    flight.plot(ax[airport_index, time_index], color="#4c78a8", alpha=0.4)
                    break

        elif phases.query('phase == "CLIMB"'):
            for airport_index, airport in enumerate(airport_codes):
                if segment := flight.takeoff(airport).next():
                    time_index = int(segment.start <= pd.Timestamp("2021-10-07 13:20Z"))
                    flight.plot(ax[airport_index, time_index], color="#f58518", alpha=0.4)
                    break

    for i, airport in enumerate(airport_codes):
        ax[i, 0].set_title(f"{airport}", loc="left", y=0.8)

    for ax_ in ax.ravel():
        ax_.add_geometries(
            [seine_river],
            crs=PlateCarree(),
            facecolor="none",
            edgecolor="#9ecae9",
            linewidth=1.5,
        )
        paris_area.set_crs(4326).to_crs(2154).plot(
            ax=ax_,
            facecolor="none",
            edgecolor="#bab0ac",
            linestyle="dotted",
        )

        ax_.set_extent((0.78, 4.06, 47.7, 49.7))

    fig.suptitle(
        "West and East configurations in Paris airports",
        fontsize=16,
        x=0.1,
        y=0.9,
        ha="left",
    )
```

---

<nav class="tutorial-nav" aria-label="Tutorial navigation">
  <a class="prev-link" href="/tutorials/visualization/"><- Trajectory visualization</a>
  <a class="next-link" href="/tutorials/declarative/">Declarative processing -></a>
</nav>
