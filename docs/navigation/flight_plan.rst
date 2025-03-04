How to infer a flight plan from a trajectory?
=============================================


.. jupyter-execute::

    from traffic.data.samples import elal747
    from traffic.data import navaids

    subset = elal747.skip("2h30min").first("2h30min")  # type: ignore

    points = [navaids[name] for name in ["KAVOS", "PEDER"]]
    m = subset.map_leaflet(highlight=dict(red=lambda f: f.aligned(points=points)))
    assert m is not None
    for p in points:
        m.add(p)
    m
