# flake8: noqa
import numpy as np
import pandas as pd
import pyproj


def _douglas_peucker_rec(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, tolerance: float
) -> None:
    l = len(x)
    if l < 3:
        return

    v = np.array([[y[len(x) - 1] - y[0]], [x[0] - x[len(x) - 1]]])
    d = np.abs(
        np.dot(
            np.dstack([x[1:-1] - x[0], y[1:-1] - y[0]])[0],
            v / np.sqrt(np.sum(v * v)),
        )
    )

    if np.max(d) < tolerance:
        mask[np.s_[1 : l - 1]] = 0
        return

    arg = np.argmax(d)
    _douglas_peucker_rec(x[: arg + 2], y[: arg + 2], mask[: arg + 2], tolerance)
    _douglas_peucker_rec(x[arg + 1 :], y[arg + 1 :], mask[arg + 1 :], tolerance)


def _douglas_peucker_rec_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    tolerance: float,
) -> None:
    l = len(x)
    if l < 3:
        return

    start = np.array([x[0], y[0], z[0]])
    end = np.array([x[-1], y[-1], z[-1]])
    point = np.dstack([x[1:], y[1:], z[1:]])[0] - start
    d = np.cross(point, (start - end) / np.linalg.norm(start - end))
    d = np.sqrt(np.sum(d * d, axis=1))

    if np.max(d) < tolerance:
        mask[np.s_[1 : l - 1]] = 0
        return

    arg = np.argmax(d)
    _douglas_peucker_rec_3d(
        x[: arg + 2], y[: arg + 2], z[: arg + 2], mask[: arg + 2], tolerance
    )
    _douglas_peucker_rec_3d(
        x[arg + 1 :], y[arg + 1 :], z[arg + 1 :], mask[arg + 1 :], tolerance
    )


def douglas_peucker(
    *args,
    df: pd.DataFrame = None,
    tolerance: float,
    x="x",
    y="y",
    z=None,
    z_factor: float = 3.048,
    lat=None,
    lon=None,
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

    if df is not None and isinstance(lat, str) and isinstance(lon, str):
        lat, lon = df[lat], df[lon]
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
        x, y = transformer.transform(lon.values, lat.values,)
    else:
        if df is not None:
            x, y = df[x].values, df[y].values
        x, y = np.array(x), np.array(y)

    if z is not None:
        if df is not None:
            z = df[z].values
        z = z_factor * np.array(z)

    mask = np.ones(len(x), dtype=bool)
    if z is None:
        _douglas_peucker_rec(x, y, mask, tolerance)
    else:
        _douglas_peucker_rec_3d(x, y, z, mask, tolerance)

    return mask
