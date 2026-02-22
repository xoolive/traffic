# Flight and Traffic objects

The `traffic` library provides methods and attributes for trajectories
and collections of trajectories, represented as pandas `DataFrame`.
A single trajectory is embedded in a `Flight` structure, a collection of trajectories in a `Traffic` structure.

## Flight objects

`Flight` is the core class offering representations, methods and attributes to single trajectories. Trajectories can either:

- **be imported from the sample trajectory set**;
- be [downloaded from The OpenSky Network](../user-guide/opensky.md);
- be loaded from a tabular file (csv, json, parquet, etc.);
- be decoded from raw ADS-B signals or streams.

We reuse here the flight introduce in our [previous page](basic.md).

```python
from traffic.data.samples import belevingsvlucht
```

Many representations are available:

If you run Python with `uv run python`:

```python
>>> print(belevingsvlucht)
Flight(icao24='484506', callsign='TRA051')
```

!!! tip "Rich representations"

    If you activate Rich representations, rendering is adapted:

    ```python
    from rich.pretty import pprint  # (1)!
    pprint(belevingsvlucht)
    ```

    1. See the official documentation <https://rich.readthedocs.io/>

    <div class="output text_html"><pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Flight</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">icao24</span>=<span style="color: #008000; text-decoration-color: #008000">'484506'</span>, <span style="color: #808000; text-decoration-color: #808000">callsign</span>=<span style="color: #008000; text-decoration-color: #008000">'TRA051'</span><span style="font-weight: bold">)</span>
    </pre>
    </div>

    ```python
    # the console is not necessary if you ran pretty.install()
    from rich.console import Console
    console = Console()
    console.print(belevingsvlucht)
    ```

    <div class="output text_html"><pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000080; text-decoration-color: #000080; font-weight: bold">Flight </span>
    - <span style="font-weight: bold">callsign:</span> TRA051
    - <span style="font-weight: bold">aircraft:</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">484506</span> · 🇳🇱 PH-HZO <span style="font-weight: bold">(</span>B738<span style="font-weight: bold">)</span>
    - <span style="font-weight: bold">start:</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2018</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">05</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span> <span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">15:21:38</span>Z
    - <span style="font-weight: bold">stop:</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2018</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">05</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span> <span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">20:22:56</span>Z
    - <span style="font-weight: bold">duration:</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> days <span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">05:01:18</span>
    - <span style="font-weight: bold">sampling rate:</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">second</span><span style="font-weight: bold">(</span>s<span style="font-weight: bold">)</span>
    - <span style="font-weight: bold">features:</span>
      o altitude, <span style="font-style: italic">int64</span>
      o groundspeed, <span style="font-style: italic">int64</span>
      o latitude, <span style="font-style: italic">double</span>
      o longitude, <span style="font-style: italic">double</span>
      o timestamp, <span style="font-style: italic">timestamp</span>
      o track, <span style="font-style: italic">int64</span>
      o vertical_rate, <span style="font-style: italic">int64</span>

  </pre>
  </div>

Specific representations are available in **Jupyter based environments** (including notebooks, Visual Studio Code interactive windows, Google Colab, etc.)

```python
belevingsvlucht
```

<div id="bv-view" class="obs-view-host"></div>

<script type="module">
import {Runtime} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/+esm";
import define from "https://api.observablehq.com/@xoolive/traffic-js.js?v=3";

const dest = document.getElementById("bv-view");
try {
  const main = new Runtime().module(define);
  const belevingsvlucht = await main.value("belevingsvlucht");
  const viewNode = await belevingsvlucht.view();
  dest.append(viewNode);
} catch (err) {
  console.error(err);
  dest.textContent = "Failed to load traffic.js demo.";
}
</script>

Information about each Flight instance are available through attributes or properties:

```python
>>> dict(belevingsvlucht)
{
    'callsign': 'TRA051',
    'icao24': '484506',
    'aircraft': Tail(icao24='484506', registration='PH-HZO', typecode='B738', flag='🇳🇱'),
    'start': Timestamp('2018-05-30 15:21:38+0000', tz='UTC'),
    'stop': Timestamp('2018-05-30 20:22:56+0000', tz='UTC'),
    'duration': Timedelta('0 days 05:01:18')
}
```

Methods are provided to select relevant parts of the flight, e.g. based on timestamps. The start and stop properties refer to the timestamps of the first and last recorded samples. Note that all timestamps are by default set to universal time (UTC) as it is common practice in aviation.

```python
>>> belevingsvlucht.start
Timestamp('2018-05-30 15:21:38+0000', tz='UTC')
>>> belevingsvlucht.stop
Timestamp('2018-05-30 20:22:56+0000', tz='UTC')
```

And also methods to modify them:

```python
belevingsvlucht.first(minutes=30)
```

<div id="bv-view-30" class="obs-view-host"></div>

<script type="module">
import {Runtime} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/+esm";
import define from "https://api.observablehq.com/@xoolive/traffic-js.js?v=3";

const dest = document.getElementById("bv-view-30");
try {
  const main = new Runtime().module(define);
  const belevingsvlucht = await main.value("belevingsvlucht");
  const ts_30min = new Date(belevingsvlucht.start.getTime() + 30 * 60 * 1000);
  const viewNode = await belevingsvlucht.before(ts_30min).view();
  dest.append(viewNode);
} catch (err) {
  console.error(err);
  dest.textContent = "Failed to load traffic.js demo.";
}
</script>

!!! warning

    Watch the difference between `strict` (`>`) and inclusive (`>=`) timestamp comparisons.

    ``` python
    >>> belevingsvlucht.after("2018-05-30 19:00").start
    Timestamp('2018-05-30 19:00:01+0000', tz='UTC')
    >>> belevingsvlucht.after("2018-05-30 19:00", strict=False).start
    Timestamp('2018-05-30 19:00:00+0000', tz='UTC')
    ```

!!! note

    Each `Flight` wraps a pandas `DataFrame`.
    If a method is missing for a specific task, access the underlying dataframe directly.

```python
belevingsvlucht.between("2018-05-30 19:00", "2018-05-30 20:00").data
```

<div id="bv-tab" class="obs-table-host"></div>

<script type="module">
import {Runtime} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/+esm";
import define from "https://api.observablehq.com/@xoolive/traffic-js.js?v=3";

const dest = document.getElementById("bv-tab");
try {
  const main = new Runtime().module(define);
  const belevingsvlucht = await main.value("belevingsvlucht");
  const start = new Date("2018-05-30 19:00Z");
  const stop = new Date("2018-05-30 20:00Z");
  const viewNode = await belevingsvlucht.after(start).before(stop).table();
  dest.append(viewNode);
} catch (err) {
  console.error(err);
  viewHost.textContent = "Failed to load traffic.js demo.";
}
</script>

## Traffic objects

Traffic is the core class to represent collections of trajectories. In practice, all trajectories are flattened in the same pd.DataFrame.

```python
from traffic.data.samples import quickstart
```

The basic representation of a Traffic object is a summary view of the data: the structure tries to infer how to separate trajectories in the data structure based on customizable heuristics, and returns a number of sample points for each trajectory.

```python
quickstart
```

<div class="output">
<h4><b>Traffic</b></h4> with 236 identifiers
<div class="traffic-summary-wrap">
<table class="traffic-summary">
  <thead>
    <tr><td>icao24</td><td>callsign</td><td>count</td></tr>
  </thead>
  <tbody>
    <tr><th>39d300</th><th>TVF91KQ</th><td class="count-bar" style="--bar: 100.0%">3893</td></tr>
    <tr><th>39b002</th><th>FHMAC</th><td class="count-bar" style="--bar: 86.3%">3360</td></tr>
    <tr><th>3aabfc</th><th>FMY8055</th><td class="count-bar" style="--bar: 68.6%">2669</td></tr>
    <tr><th>39c82b</th><th>PEA501</th><td class="count-bar" style="--bar: 57.7%">2247</td></tr>
    <tr><th>4241bb</th><th>VPCAL</th><td class="count-bar" style="--bar: 55.7%">2168</td></tr>
    <tr><th>02a195</th><th>TAR722</th><td class="count-bar" style="--bar: 55.6%">2166</td></tr>
    <tr><th>398495</th><th>CCM774V</th><td class="count-bar" style="--bar: 54.8%">2134</td></tr>
    <tr><th>4bc844</th><th>PGT90Y</th><td class="count-bar" style="--bar: 54.6%">2124</td></tr>
    <tr><th>39ceb4</th><th>TVF19YP</th><td class="count-bar" style="--bar: 53.3%">2076</td></tr>
    <tr><th>4d02be</th><th>JFA12P</th><td class="count-bar" style="--bar: 52.8%">2057</td></tr>
  </tbody>
</table>
</div>
</div>

Traffic objects offer the ability to index and iterate on all flights contained in the structure.
In order to separate and identify trajectories (Flight), Traffic objects will use either:

- a customizable flight identifier (flight_id); or
- a combination of timestamp and icao24 (aircraft identifier);

Indexation will be made on:

- icao24, callsign (or flight_id if available):

      ```python
      quickstart["02a195"]  # (1)!
      quickstart["TAR722"]  # (2)!
      ```

      1. return type: Flight, based on icao24
      2. return type: Flight, based on callsign

      <div id="tar722" class="obs-view-host"></div>

<script type="module">
import {Runtime} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@5/+esm";
import define from "https://api.observablehq.com/@xoolive/traffic-js.js?v=3";

const dest = document.getElementById("tar722");
try {
  const main = new Runtime().module(define);
  const quickstart = await main.value("quickstart");
  const flight = Array.from(quickstart).filter((flight) => flight.callsign === "TAR722")[0];
  const v = await flight.view();
  dest.append(v);
} catch (err) {
  console.error(err);
  viewHost.textContent = "Failed to load traffic.js demo.";
}
</script>

- an integer or a slice, to take flights in order in the collection:

      ```python
      quickstart[0]  # (1)!
      quickstart[:10]  # (2)!
      ```

      1. return the first trajectory in the collection: that's a Flight
      2. return the 10 first trajectories in the collection: that's a Traffic

    <div class="output">
      <h4><b>Traffic</b></h4> with 10 identifiers
      <div class="traffic-summary-wrap">
      <table class="traffic-summary">
        <thead>
          <tr><td>icao24</td><td>callsign</td><td>count</td></tr>
        </thead>
        <tbody>
          <tr><th>02a195</th><th>TAR722</th><td class="count-bar" style="--bar: 100.0%">2166</td></tr>
          <tr><th>0a0046</th><th>DAH1011</th><td class="count-bar" style="--bar: 61.1%">1323</td></tr>
          <tr><th>0101de</th><th>MSR799</th><td class="count-bar" style="--bar: 59.6%">1290</td></tr>
          <tr><th>34150e</th><th>IBE34AK</th><td class="count-bar" style="--bar: 53.2%">1152</td></tr>
          <tr><th>06a2b1</th><th>QTR9UU</th><td class="count-bar" style="--bar: 52.8%">1144</td></tr>
          <tr><th>0a0047</th><th>DAH1000</th><td class="count-bar" style="--bar: 51.3%">1111</td></tr>
          <tr><th>300789</th><th>IWALK</th><td class="count-bar" style="--bar: 47.3%">1024</td></tr>
          <tr><th>06a1e7</th><th>QTR23JR</th><td class="count-bar" style="--bar: 40.4%">874</td></tr>
          <tr><th>06a133</th><th>QQE940</th><td class="count-bar" style="--bar: 37.1%">803</td></tr>
          <tr><th>0a0047</th><th>DAH1001</th><td class="count-bar" style="--bar: 36.8%">798</td></tr>
        </tbody>
      </table>
      </div>
      </div>

- a subset of trajectories can also be selected with a list:

  ```python
  quickstart[["AFR83HQ", "AFR83PX", "AFR84UW", "AFR91QD"]]
  ```

  <div class="output">
  <h4><b>Traffic</b></h4> with 4 identifiers
  <div class="traffic-summary-wrap">
  <table class="traffic-summary">
    <thead>
      <tr><td>icao24</td><td>callsign</td><td>count</td></tr>
    </thead>
    <tbody>
      <tr><th>394c04</th><th>AFR83PX</th><td class="count-bar" style="--bar: 100.0%">1274</td></tr>
      <tr><th>3946e2</th><th>AFR84UW</th><td class="count-bar" style="--bar: 87.3%">1112</td></tr>
      <tr><th>3946e0</th><th>AFR91QD</th><td class="count-bar" style="--bar: 84.6%">1078</td></tr>
      <tr><th>3950d0</th><th>AFR83HQ</th><td class="count-bar" style="--bar: 51.0%">650</td></tr>
    </tbody>
  </table>
  </div>
  </div>

- or with a pandas-like query():

  ```python
  quickstart.query('callsign.str.startswith("AFR")')
  ```

  <div class="output">
  <h4><b>Traffic</b></h4> with 84 identifiers
  <div class="traffic-summary-wrap">
  <table class="traffic-summary">
    <thead>
      <tr><td>icao24</td><td>callsign</td><td>count</td></tr>
    </thead>
    <tbody>
      <tr><th>393324</th><th>AFR69CR</th><td class="count-bar" style="--bar: 100.0%">1992</td></tr>
      <tr><th>393320</th><th>AFR85FF</th><td class="count-bar" style="--bar: 99.1%">1975</td></tr>
      <tr><th>398564</th><th>AFR9455</th><td class="count-bar" style="--bar: 83.6%">1666</td></tr>
      <tr><th>3985a4</th><th>AFR19BH</th><td class="count-bar" style="--bar: 82.1%">1636</td></tr>
      <tr><th>3944f1</th><th>AFR15AH</th><td class="count-bar" style="--bar: 77.6%">1546</td></tr>
      <tr><th>3944f0</th><th>AFR51LU</th><td class="count-bar" style="--bar: 76.6%">1525</td></tr>
      <tr><th>393321</th><th>AFR18KJ</th><td class="count-bar" style="--bar: 74.6%">1486</td></tr>
      <tr><th>3944ed</th><th>AFR71ZP</th><td class="count-bar" style="--bar: 73.4%">1463</td></tr>
      <tr><th>3950cd</th><th>AFR26TR</th><td class="count-bar" style="--bar: 70.1%">1396</td></tr>
      <tr><th>394c13</th><th>AFR1753</th><td class="count-bar" style="--bar: 67.6%">1346</td></tr>
    </tbody>
  </table>
  </div>
  </div>

---

<nav class="tutorial-nav" aria-label="Tutorial navigation">
  <a class="prev-link" href="/tutorials/"><- Overview</a>
  <a class="next-link" href="/tutorials/visualization/">Trajectory visualization -></a>
</nav>
