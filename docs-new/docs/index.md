---
title: ""
hide: title
---

<p align="center">
  <img
    src="assets/logo/logo_full.png"
    alt="traffic -- air traffic data processing with Python"
    width="560"
  />
</p>

<p class="hero-actions" align="center">
  <a class="md-button md-button--primary" href="getting-started/">Get Started</a>
  <a class="md-button" href="reference/">API Reference</a>
</p>

`traffic` combines a **high-level trajectory API** with direct [pandas](https://pandas.pydata.org/) access: **fast to explore**, precise when you need control.

It connects to **open ADS-B and operational sources**, e.g. [OpenSky Network](https://opensky-network.org/), Eurocontrol, FAA, and custom datasets, so you can go from raw records to reproducible analysis quickly.

## What traffic is about

<div class="home-grid">
  <div class="home-card">
    <h3>Trajectory Analysis</h3>
    <p>Manipulate <code>Flight</code> and <code>Traffic</code> structures to filter, segment, or enrich trajectories.</p>
  </div>
  <div class="home-card">
    <h3>Operational Algorithms</h3>
    <p>Use built-in or implement metadata, navigation, ground movement, clustering or prediction algorithms.</p>
  </div>
  <div class="home-card">
    <h3>Data Integration</h3>
    <p>Connect OpenSky Network, Eurocontrol or FAA data, and custom datasets through a consistent Python API.</p>
  </div>
  <div class="home-card">
    <h3>Visual Exploration</h3>
    <p>Render maps and timelines with Matplotlib, Altair, Plotly and Leaflet-based workflows.</p>
  </div>
</div>

## Basic example

!!!note "Belevingsvlucht (2018)"

    In 2018, the Dutch government organised a _“belevingsvlucht”_ (“experience flight”) to let policymakers experience the planned low-altitude flight routes for Lelystad Airport. The intention was to demonstrate that the routes were acceptable in practice. Instead, the flight highlighted how complex and constrained these trajectories were, reinforcing concerns about noise impact, airspace complexity, and operational realism.

The library provides few sample trajectories for illustration purposes.

```python
from traffic.data import airports
from traffic.data.samples import belevingsvlucht

m = belevingsvlucht.map_leaflet(
    zoom=7,
    highlight={"orange": "first('15 min')", "red": "holding_pattern"},
)

m.add(airports["EHLE"].point.leaflet(title="Lelystad Airport"))
m.add(airports["EHAM"].point.leaflet(title="Amsterdam Airport Schiphol"))

m
```

<div id="map" style="height: 400px;"></div>

<script type="module">
import {Runtime} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/+esm";
import define from "https://api.observablehq.com/@xoolive/traffic-js.js?v=3";

const container = document.getElementById("map");
container.textContent = "Loading trajectory...";

try {
  const main = new Runtime().module(define);

  const [belevingsvlucht, airports] = await Promise.all([
    main.value("belevingsvlucht"),
    main.value("airports")
  ]);

  container.textContent = "";

  const map = L.map(container, {scrollWheelZoom: false});

  const layer = L.geoJSON(belevingsvlucht.feature(), {
    style: {
      color: "#2563eb",
      weight: 3,
      opacity: 0.9
    }
  }).addTo(map);

  const ts_15min = new Date(belevingsvlucht.start.getTime() + 15 * 60 * 1000);
  L.geoJSON(belevingsvlucht.before(ts_15min).feature(), {
    style: {
      color: "orange",
      weight: 3,
      opacity: 1
    }
  }).addTo(map);

  // for now...
  const hp_start = new Date("2018-05-30 15:43:52Z")
  const hp_stop = new Date("2018-05-30 15:53:51Z")
  L.geoJSON(belevingsvlucht.before(hp_stop).after(hp_start).feature(), {
    style: {
      color: "red",
      weight: 3,
      opacity: 1
    }
  }).addTo(map);

  map.fitBounds(layer.getBounds(), {maxZoom: 10});

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
  }).addTo(map);

  for (const code of ["EHAM", "EHLE"]) {
    const ap = airports[code];
    if (!ap) continue;
    L.marker([ap.latitude, ap.longitude])
      .addTo(map)
      .bindTooltip(ap.name, /*{permanent: code === "EHLE"}*/);
  }
} catch (err) {
  console.error(err);
  container.innerHTML = "<p>Unable to load Observable module data for this demo.</p>";
}


</script>

## Contents

- [Installation](installation.md)
- [Getting Started](getting-started.md)
- [Tutorials](tutorials/index.md)
- [User Guide](user-guide/index.md)
- [Reference](reference/index.md)
- [Publications](publications.md)
