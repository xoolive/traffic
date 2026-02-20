# Installation

This project is now centered on [`uv`](https://docs.astral.sh/uv/) for local
development, reproducibility, and CI parity.

!!! warning

    The former Conda-based installation path is being deprecated. Use `uv` as the default workflow.

## Prerequisites

- `uv` installed on your machine: <https://docs.astral.sh/uv/>
- Python 3.11+, which can also be installed through uv

## Quick install (recommended)

- If you want to include traffic in your project:

  ```bash
  uv add traffic
  ```

- If you just work in the traffic repository:

  ```bash
  git clone --depth 1 https://github.com/xoolive/traffic
  cd traffic
  uv sync
  ```

Run commands with:

```bash
uv run <command>
```

For example:

```bash
uv run python -c "import traffic; print(traffic.__version__)"
```

## Extras (advanced)

The project keeps optional features as extras to stay lightweight by default.
This reduces install size and avoids pulling heavy stacks when not needed.

Available extras in this repository:

- `altair`
- `plotly`
- `leaflet`
- `lonboard` (Python < 3.13)
- `spark` (includes `pyspark`, heavy)
- `xarray`
- `learning` (scikit-learn)

Install only what you need, for example:

```bash
uv sync --extra altair --extra leaflet
```

!!! note

    `--all-extras` is convenient but often excessive for day-to-day work, especially because it pulls heavy dependencies like `pyspark`.

## Updating

To refresh your environment from lock/config updates:

```bash
uv sync --dev
```

## Required quality checks before opening a PR

Install git hooks with [`prek`](https://prek.readthedocs.io/) (replacing pre-commit). Install `prek` following its own installation documentation, then run:

```bash
prek install
```

!!! note

    `prek` is not installed through this project's `uv sync` environment by default.

Run formatting, linting, and typing locally:

```bash
uv run ruff format src tests scripts
uv run ruff check src tests scripts
uv run ty check src tests scripts
```

Tool references:

- [`uv` documentation](https://docs.astral.sh/uv/)
- [`ruff` documentation](https://docs.astral.sh/ruff/) (formatting, linting)
- [`ty` documentation](https://github.com/astral-sh/ty) (static analysis)

Then run tests (at least the fast/local subset you touched):

```bash
uv run pytest
```
