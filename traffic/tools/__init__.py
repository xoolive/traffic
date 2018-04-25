try:
    from .bresenham import bresenham, bresenham_multiply
except ImportError:
    import warnings
    warnings.warn("No compiled version found. Cythonizing now...")
    import pyximport
    pyximport.install()
    from .bresenham import bresenham, bresenham_multiply  # noqa

from .douglas_peucker import douglas_peucker
