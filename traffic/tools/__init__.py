try:
    from .bresenham import bresenham
except ImportError:
    import warnings
    warnings.warn("No compiled version found. Cythonizing now...")
    import pyximport
    pyximport.install()
    from .bresenham import bresenham  # noqa
