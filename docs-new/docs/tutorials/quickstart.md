The `traffic` library provides methods and attributes for trajectories
and collections of trajectories, represented as pandas `DataFrame`.
A single trajectory is embedded in a `Flight` structure, a collection of trajectories in a `Traffic` structure.

## Flight objects

`Flight` is the core class offering representations, methods and attributes to single trajectories. Trajectories can either:

- **be imported from the sample trajectory set**;
- be downloaded from The OpenSky Network;
- be loaded from a tabular file (csv, json, parquet, etc.);
- be decoded from raw ADS-B signals.

!!!note "Belevingsvlucht (2018)"

    In 2018, the Dutch government organised a _“belevingsvlucht”_ (“experience flight”) to let policymakers experience the planned low-altitude flight routes for Lelystad Airport. The intention was to demonstrate that the routes were acceptable in practice. Instead, the flight highlighted how complex and constrained these trajectories were, reinforcing concerns about noise impact, airspace complexity, and operational realism.

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

    If you activate `rich` representations, per <https://rich.readthedocs.io/>, rendering is adapted:

    ```python
    from rich.pretty import pprint
    pprint(belevingsvlucht)
    ```

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
  viewHost.textContent = "Failed to load traffic.js demo.";
}
</script>

Flight objects have a number of attributes:

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
>>> belevingsvlucht.start
Timestamp('2018-05-30 15:21:38+0000', tz='UTC')
>>> belevingsvlucht.stop
Timestamp('2018-05-30 20:22:56+0000', tz='UTC')
```

And also methods to modify them

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
  viewHost.textContent = "Failed to load traffic.js demo.";
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

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

</div>

## Traffic objects

```python
from traffic.data.samples import quickstart
```

```python
quickstart
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

<h4><b>Traffic</b></h4> with 236 identifiers<style type="text/css">
#T_901fc_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
#T_901fc_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 86.3%, transparent 86.3%);
}
#T_901fc_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 68.6%, transparent 68.6%);
}
#T_901fc_row3_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 57.7%, transparent 57.7%);
}
#T_901fc_row4_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 55.7%, transparent 55.7%);
}
#T_901fc_row5_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 55.6%, transparent 55.6%);
}
#T_901fc_row6_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 54.8%, transparent 54.8%);
}
#T_901fc_row7_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 54.6%, transparent 54.6%);
}
#T_901fc_row8_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 53.3%, transparent 53.3%);
}
#T_901fc_row9_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 52.8%, transparent 52.8%);
}
</style>

|        |          | count |
| ------ | -------- | ----- |
| icao24 | callsign |       |
| 39d300 | TVF91KQ  | 3893  |
| 39b002 | FHMAC    | 3360  |
| 3aabfc | FMY8055  | 2669  |
| 39c82b | PEA501   | 2247  |
| 4241bb | VPCAL    | 2168  |
| 02a195 | TAR722   | 2166  |
| 398495 | CCM774V  | 2134  |
| 4bc844 | PGT90Y   | 2124  |
| 39ceb4 | TVF19YP  | 2076  |
| 4d02be | JFA12P   | 2057  |

```python
quickstart["TAR722"]
quickstart["39b002"]
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

![](quickstart_files/figure-commonmark/cell-11-output-2.svg)

```python
quickstart[0]
quickstart[:10]
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

<h4><b>Traffic</b></h4> with 10 identifiers<style type="text/css">
#T_9443a_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
#T_9443a_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 61.1%, transparent 61.1%);
}
#T_9443a_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 59.6%, transparent 59.6%);
}
#T_9443a_row3_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 53.2%, transparent 53.2%);
}
#T_9443a_row4_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 52.8%, transparent 52.8%);
}
#T_9443a_row5_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 51.3%, transparent 51.3%);
}
#T_9443a_row6_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 47.3%, transparent 47.3%);
}
#T_9443a_row7_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 40.4%, transparent 40.4%);
}
#T_9443a_row8_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 37.1%, transparent 37.1%);
}
#T_9443a_row9_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 36.8%, transparent 36.8%);
}
</style>

|        |          | count |
| ------ | -------- | ----- |
| icao24 | callsign |       |
| 02a195 | TAR722   | 2166  |
| 0a0046 | DAH1011  | 1323  |
| 0101de | MSR799   | 1290  |
| 34150e | IBE34AK  | 1152  |
| 06a2b1 | QTR9UU   | 1144  |
| 0a0047 | DAH1000  | 1111  |
| 300789 | IWALK    | 1024  |
| 06a1e7 | QTR23JR  | 874   |
| 06a133 | QQE940   | 803   |
| 0a0047 | DAH1001  | 798   |

```python
quickstart[["AFR83HQ", "AFR83PX", "AFR84UW", "AFR91QD"]]
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

<h4><b>Traffic</b></h4> with 4 identifiers<style type="text/css">
#T_f35a2_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
#T_f35a2_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 87.3%, transparent 87.3%);
}
#T_f35a2_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 84.6%, transparent 84.6%);
}
#T_f35a2_row3_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 51.0%, transparent 51.0%);
}
</style>

|        |          | count |
| ------ | -------- | ----- |
| icao24 | callsign |       |
| 394c04 | AFR83PX  | 1274  |
| 3946e2 | AFR84UW  | 1112  |
| 3946e0 | AFR91QD  | 1078  |
| 3950d0 | AFR83HQ  | 650   |

```python
quickstart.query('callsign.str.startswith("AFR")')
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>

<h4><b>Traffic</b></h4> with 84 identifiers<style type="text/css">
#T_dc55d_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
#T_dc55d_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 99.1%, transparent 99.1%);
}
#T_dc55d_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 83.6%, transparent 83.6%);
}
#T_dc55d_row3_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 82.1%, transparent 82.1%);
}
#T_dc55d_row4_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 77.6%, transparent 77.6%);
}
#T_dc55d_row5_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 76.6%, transparent 76.6%);
}
#T_dc55d_row6_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 74.6%, transparent 74.6%);
}
#T_dc55d_row7_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 73.4%, transparent 73.4%);
}
#T_dc55d_row8_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 70.1%, transparent 70.1%);
}
#T_dc55d_row9_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 67.6%, transparent 67.6%);
}
</style>

|        |          | count |
| ------ | -------- | ----- |
| icao24 | callsign |       |
| 393324 | AFR69CR  | 1992  |
| 393320 | AFR85FF  | 1975  |
| 398564 | AFR9455  | 1666  |
| 3985a4 | AFR19BH  | 1636  |
| 3944f1 | AFR15AH  | 1546  |
| 3944f0 | AFR51LU  | 1525  |
| 393321 | AFR18KJ  | 1486  |
| 3944ed | AFR71ZP  | 1463  |
| 3950cd | AFR26TR  | 1396  |
| 394c13 | AFR1753  | 1346  |
