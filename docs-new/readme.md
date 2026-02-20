# docs-new

Zensical-first documentation scaffold for the `traffic` project.

This folder is intentionally independent from the current Sphinx docs in `docs/`.
It lets us migrate pages incrementally while keeping existing docs live.

## Local usage

Preferred path (no local install required):

```bash
uv run zensical serve -f docs-new/zensical.yml
```

Build static output:

```bash
uv run zensical build -f docs-new/zensical.yml
```

Generate publications page from BibTeX before build:

```bash
uv run python docs-new/tools/generate_publications.py
uv run zensical build -f docs-new/zensical.yml
```

Quarto pilot page (quickstart):

```bash
uvx quarto render docs-new/quarto/quickstart.qmd
```

Compatibility fallback:

```bash
uv run mkdocs serve -f docs-new/zensical.yml
```

## Migration policy (nutshell)

- `docs/` remains source of truth until parity is reached.
- New/ported pages go to `docs-new/docs/`.
- API reference pages are markdown-first placeholders for now; later we can
  wire mkdocstrings (or generated API pages).
- Notebook-heavy pages are tracked in `docs-new/docs/tutorials/` as candidates
  for Quarto-backed rendering.
