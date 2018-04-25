# cython: embedsignature=False
cimport cython

@cython.boundscheck(False)
cdef int grid_increment(long x0, long y0, long x1, long y1, long mult,
                        long[:, :] grid):

    cdef unsigned nrows, ncols
    cdef long e2, sx, sy, err
    cdef long dx, dy

    nrows = grid.shape[0]
    ncols = grid.shape[1]

    dx = x1 - x0 if x1 > x0 else x0 - x1
    dy = y1 - y0 if y1 > y0 else y0 - y1

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        # When endpoint is 0, this test occurs before we increment the
        # grid value, so we don't count the last point.
        if x0 == x1 and y0 == y1:
            break

        if (0 <= x0 < nrows) and (0 <= y0 < ncols):
            grid[x0, y0] += mult

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return 0


def bresenham(long[:, :] points, long[:, :] grid):
    """bresenham(long[:, :] points, long[:, :] grid)
    Bresenham's algorithm.

    - points is a memory view over a 2D numpy array (the trajectory)
    - grid is a memory view over a 2D numpy array (the grid)

    The algorithms adds 1 to all the cells the trajectory cross.

    >>> import numpy as np
    >>> points = np.array([[0, 0], [1, 11]], dtype=np.int64)
    >>> grid = np.zeros((2, 11), dtype=np.int64)
    >>> bresenham(points, grid)
    >>> grid
    array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    See also: http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """

    cdef unsigned k = 0
    cdef int x1 = 0, y1 = 0

    with cython.boundscheck(False):
        for k in range(points.shape[0] - 1):
            x1 = points[k+1, 0]
            y1 = points[k+1, 1]
            grid_increment(points[k, 0], points[k, 1], x1, y1, 1, grid)

        if 0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]:
            # Count the last point in the curve.
            grid[x1, y1] += 1


def bresenham_multiply(long[:, :] points, long[:] mult, long[:, :] grid):
    """bresenham(long[:, :] points, long[:] mult, long[:, :] grid)
    Bresenham's algorithm.

    - points is a memory view over a 2D numpy array (the trajectory)
    - mult is a memory view over a 1D numpy array (vertrate, bearing, etc.)
    - grid is a memory view over a 2D numpy array (the grid)

    The algorithms adds 1 to all the cells the trajectory cross.

    >>> import numpy as np
    >>> points = np.array([[0, 0], [1, 11]], dtype=np.int64)
    >>> grid = np.zeros((2, 11), dtype=np.int64)
    >>> bresenham(points, grid)
    >>> grid
    array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    See also: http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """

    cdef unsigned k = 0
    cdef int x1 = 0, y1 = 0

    with cython.boundscheck(False):
        for k in range(points.shape[0] - 1):
            x1 = points[k+1, 0]
            y1 = points[k+1, 1]
            grid_increment(points[k, 0], points[k, 1], x1, y1, mult[k], grid)

        if 0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]:
            # Count the last point in the curve.
            grid[x1, y1] += mult[k]

