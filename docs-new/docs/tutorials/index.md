# Tutorials

Tutorials are likely the strongest Quarto candidates because they are:

- long narrative flows,
- execution-heavy,
- visual and notebook-oriented.

## Current candidates

- `docs/tutorial.rst`
- `docs/tutorial/part3_occupancy.ipynb`
- `docs/tutorial/part4_citypair.ipynb`
- `docs/statistical/*.rst`
- selected `docs/paper/*.rst`

## Migration policy

- Keep concise how-to guides in MkDocs markdown.
- Move notebook-style or research-style walkthroughs to Quarto-backed pages.

# Getting started

This page is the Quarto pilot migration of `docs/quickstart.rst`.

It keeps the original narrative and executable style while replacing
Sphinx directives with notebook-native cells.

## Scope of this pilot

We keep the same major blocks as the original page:

1.  basic introduction to `Flight` and `Traffic`
2.  visualizations of trajectory data
3.  low-altitude trajectory patterns around Paris
4.  declarative trajectory processing through lazy iteration

> [!TIP]
>
> This Quarto pilot is meant to validate structure and rendering first.
> We can incrementally re-introduce all code cells from the original RST
> page.
