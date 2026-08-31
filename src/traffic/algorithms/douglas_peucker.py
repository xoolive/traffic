from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj


def _douglas_peucker_iter(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    tolerance: float,
) -> None:
    # Iterative Ramer-Douglas-Peucker (explicit stack) to avoid the
    # RecursionError the recursive formulation triggered on long
    # trajectories, notably on Python 3.14 (see issue #568).
    n = len(x)
    if n < 3:
        return
    stack: list[tuple[int, int]] = [(0, n)]
    while stack:
        start, end = stack.pop()
        len_seg = end - start
        if len_seg < 3:
            continue
        # Slices are views into the original arrays, so writing to
        # mask_seg mutates ``mask`` at the right absolute positions.
        x_seg = x[start:end]
        y_seg = y[start:end]
        mask_seg = mask[start:end]

        v = np.array([[y_seg[-1] - y_seg[0]], [x_seg[0] - x_seg[-1]]])
        d = np.abs(
            np.dot(
                np.dstack([x_seg[1:-1] - x_seg[0], y_seg[1:-1] - y_seg[0]])[0],
                v / np.sqrt(np.sum(v * v)),
            )
        )

        if np.max(d) < tolerance:
            mask_seg[np.s_[1 : len_seg - 1]] = 0
            continue

        arg = cast(int, np.argmax(d))
        # The farthest point sits at segment index ``arg + 1``; the two
        # sub-ranges mirror the recursive ``x[: arg + 2]`` / ``x[arg + 1:]``
        # splits and share that point.
        farthest = start + arg + 1
        stack.append((start, farthest + 1))
        stack.append((farthest, end))


def _douglas_peucker_iter_3d(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    z: npt.NDArray[Any],
    mask: npt.NDArray[Any],
    tolerance: float,
) -> None:
    # Iterative 3D variant; see _douglas_peucker_iter for the rationale.
    n = len(x)
    if n < 3:
        return
    stack: list[tuple[int, int]] = [(0, n)]
    while stack:
        start, end = stack.pop()
        len_seg = end - start
        if len_seg < 3:
            continue
        x_seg = x[start:end]
        y_seg = y[start:end]
        z_seg = z[start:end]
        mask_seg = mask[start:end]

        start_pt = np.array([x_seg[0], y_seg[0], z_seg[0]])
        end_pt = np.array([x_seg[-1], y_seg[-1], z_seg[-1]])
        point = np.dstack([x_seg[1:], y_seg[1:], z_seg[1:]])[0] - start_pt
        d = np.cross(
            point, (start_pt - end_pt) / np.linalg.norm(start_pt - end_pt)
        )
        d = np.sqrt(np.sum(d * d, axis=1))

        if np.max(d) < tolerance:
            mask_seg[np.s_[1 : len_seg - 1]] = 0
            continue

        arg = cast(int, np.argmax(d))
        farthest = start + arg + 1
        stack.append((start, farthest + 1))
        stack.append((farthest, end))


def douglas_peucker(
    *,
    df: pd.DataFrame | None = None,
    tolerance: float,
    x: Union[str, pd.Series] = "x",
    y: Union[str, pd.Series] = "y",
    z: Union[None, str, pd.Series] = None,
    z_factor: float = 3.048,
    lat: Union[None, str, pd.Series] = None,
    lon: Union[None, str, pd.Series] = None,
) -> np.ndarray:
    """Ramer-Douglas-Peucker algorithm for 2D/3D trajectories.

    Simplify a trajectory by keeping the points further away from the straight
    line.

    Parameters:
        df        Optional                a Pandas dataframe
        tolerance float                   the threshold for cutting the
                                          trajectory
        z_factor  float                   for ft/m conversion (default 3.048)
                                            1km lateral, 100m vertical seems
                                            like a good ratio
        x, y, z   str or ndarray[float]   the column names if a dataframe is
                                          given, otherwise a series of float
        lat, lon  str or ndarray[float]   the column names if a dataframe is
                                          given, otherwise a series of float.
                                          x, y are built with a Lambert
                                          Conformal projection

        Note that lat, lon has precedence over x, y

    Returns:
        a np.array of booleans serving as a mask on the dataframe or
        on the numpy array

    See also: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

    """

    if df is None and (isinstance(x, str) or isinstance(y, str)):
        raise ValueError("Provide a dataframe if x and y are column names")
    if df is None and (isinstance(lon, str) or isinstance(lat, str)):
        raise ValueError("Provide a dataframe if lat and lon are column names")
    if tolerance < 0:
        raise ValueError("tolerance must be a positive float")

    x_arr: npt.NDArray[Any]
    y_arr: npt.NDArray[Any]
    z_arr: npt.NDArray[Any] | None = None

    if df is not None and isinstance(lat, str) and isinstance(lon, str):
        lat, lon = df[lat], df[lon]
    if isinstance(lat, str) or isinstance(lon, str):
        raise ValueError("lat and lon must now be Pandas Series")
    if df is not None and lat is not None and lon is not None:
        projection = pyproj.Proj(
            proj="lcc",
            ellps="WGS84",
            lat_1=lat.min(),
            lat_2=lat.max(),
            lat_0=lat.mean(),
            lon_0=lon.mean(),
        )

        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj("epsg:4326"), projection, always_xy=True
        )
        x_t, y_t = transformer.transform(lon.values, lat.values)
        x_arr = np.array(x_t)
        y_arr = np.array(y_t)
    else:
        if df is not None:
            x, y = df[x].values, df[y].values
        x_arr, y_arr = np.array(x), np.array(y)

    if z is not None:
        if df is not None:
            z = df[z].values
        z_arr = z_factor * np.array(z)

    mask: npt.NDArray[Any] = np.ones(len(x_arr), dtype=bool)
    if z_arr is None:
        _douglas_peucker_iter(x_arr, y_arr, mask, tolerance)
    else:
        _douglas_peucker_iter_3d(x_arr, y_arr, z_arr, mask, tolerance)

    return mask
