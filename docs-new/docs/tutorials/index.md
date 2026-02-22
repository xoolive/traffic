# Overview

!!!warning "Disclaimer"

    These pages are not a formal training course, and they are not meant to be full end-to-end workflows.

    Think of them as guided demos that help you quickly understand the library's philosophy: trajectories are first-class objects, operations stay composable, and exploration should remain readable.

## A basic example

!!!note "Belevingsvlucht (2018)"

    In 2018, the Dutch government organised a _“belevingsvlucht”_ (“experience flight”) to let policymakers experience the planned low-altitude flight routes for Lelystad Airport. The intention was to demonstrate that the routes were acceptable in practice. Instead, the flight highlighted how complex and constrained these trajectories were, reinforcing concerns about noise impact, airspace complexity, and operational realism.

The library provides sample trajectories for illustration purposes.

```python
from traffic.data.samples import belevingsvlucht  # (1)!
from traffic.data import airports  # (2)!

m = belevingsvlucht.map_leaflet(  # (3)!
    zoom=7,
    highlight={
      "orange": "first('15 min')",  # (4)!
      "red": "holding_pattern"  # (5)!
    },
)

m.add(airports["EHLE"].point.leaflet(title="Lelystad Airport"))  # (6)!
m.add(airports["EHAM"].point.leaflet(title="Amsterdam Airport Schiphol"))

m
```

1. You may import sample trajectories from `traffic.data.samples`.
2. Basic metadata (incl. airports) are also available.
3. The `.map_leaflet()` method offers to open a Leaflet widget in Jupyter-like interactive environments
4. Read as _"highlight in orange the first 15 minutes"_
5. Read as _"highlight in red anything looking like a holding pattern"_
6. Note the `.point` attribute, that triggers a `Marker` widget in Leaflet.

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

---

<nav class="tutorial-nav" aria-label="Tutorial navigation">
  <a class="next-link" href="/tutorials/quickstart/">Flight and Traffic objects -></a>
</nav>
