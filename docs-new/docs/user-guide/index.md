# User Guide

The User Guide is organized by task: where data comes from, how to clean and
enrich trajectories, how to detect navigation events, and how to analyze or
visualize results.

It replaces the legacy `docs/user_guide.rst` + related `data_sources/`,
`navigation/`, `statistical/`, and `visualize/` sections.

## Start here

- [Quickstart](/tutorials/quickstart/): first end-to-end workflow.
- [Holding patterns](/user-guide/holding-patterns/): first migrated deep-dive page.

## Guide map (migration status)

- **Data sources**
  - `sample-trajectories`, `airports`, `aircraft`, `navigation`, `airspace`, `flight-plans`, `opensky`, `eurocontrol`, `raw-data`
- **Processing**
  - `cleaning`, `interpolation`, `resampling`, `simplification`, `arithmetic`, `iteration`
- **Navigation events**
  - `flight-phases`, `go-around`, `runway-detection`, `holding-patterns`, `point-merge`, `top-of-climb`, `top-of-descent`
- **Analysis and modeling**
  - `cpa`, `clustering`, `prediction`, `airspace-occupancy`, `atc-deconfliction`, `emissions`
- **Visualization and operations**
  - `plotting`, `troubleshooting`, `gnss-interferance`

Most pages above are currently placeholders while content is being ported.
When in doubt, use [Quickstart](/tutorials/quickstart/) and [Tutorials](/tutorials/)
for working examples.
