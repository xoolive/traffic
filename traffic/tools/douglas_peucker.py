import numpy as np
import pandas as pd
import pyproj

def _douglas_peucker_rec(x: np.ndarray, y: np.ndarray,
                         mask: np.ndarray, tolerance: float) -> None:
    l = len(x)
    if l < 3: return

    v = np.array([[y[len(x)-1]-y[0]], [x[0] - x[len(x) -1]]])
    d = np.abs(np.dot(np.dstack([x[1:-1]-x[0], y[1:-1]-y[0]])[0],
                      v/np.sqrt(np.sum(v*v))))

    if np.max(d) < tolerance:
        mask[np.s_[1:l-1]] = 0
        return

    arg = np.argmax(d)
    _douglas_peucker_rec(x[:arg+2], y[:arg+2], mask[:arg+2], tolerance)
    _douglas_peucker_rec(x[arg+1:], y[arg+1:], mask[arg+1:], tolerance)


def douglas_peucker(*args, df=None, tolerance, x='x', y='y', lat=None, lon=None):
    """Ramer-Douglas-Peucker algorithm for 2D lines.

    Simplify a 2D-line trajectory by keeping the points further away from the
    straight-line.

    Parameters:
        df        Optional                a Pandas dataframe
        tolerance float                   the threshold for cutting the
                                          trajectory
        x, y      str or ndarray[float]   the column names if a dataframe is
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
        lat, lon = df[lat].values, df[lon].values
    if df is not None and lat is not None and lon is not None:
        lat, lon = np.array(lat), np.array(lon)
        x, y = pyproj.transform(
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(proj='lcc',
                        lat0=lat.mean(), lon0=lon.mean(),
                        lat1=lat.min(), lat2=lat.max(),
                        ), lon, lat)
    else:
        x, y = np.array(x), np.array(y)

    mask = np.ones(len(x), dtype=bool)
    _douglas_peucker_rec(x, y, mask, tolerance)

    return mask
