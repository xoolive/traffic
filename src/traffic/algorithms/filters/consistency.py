from __future__ import annotations

import operator
import warnings
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import FilterBase

NM2METER = 1852


def distance(
    lat1: npt.NDArray[np.float64],
    lon1: npt.NDArray[np.float64],
    lat2: npt.NDArray[np.float64],
    lon2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r = 6371000
    phi1 = lat1
    phi2 = lat2
    delta_phi = lat2 - lat1
    delta_lambda = lon2 - lon1
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return res / NM2METER  # type: ignore


def lag(horizon: int, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    res = np.empty((horizon, v.shape[0]))
    res.fill(np.nan)
    for i in range(horizon):
        res[i, : v.shape[0] - i] = v[i:]
    return res


def diffangle(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    d = a - b
    return d + 2 * np.pi * (  # type: ignore
        (d < -np.pi).astype(float) - (d >= np.pi).astype(float)
    )


def dxdy_from_dlat_dlon(
    lat_rad: npt.NDArray[np.float64],
    lon_rad: npt.NDArray[np.float64],
    dlat: npt.NDArray[np.float64],
    dlon: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    a = 6378137
    b = 6356752.314245
    e2 = 1 - (b / a) ** 2
    sinmu2 = np.sin(lat_rad) ** 2
    rm = a * (1 - e2) / (1 - e2 * sinmu2) ** (3 / 2)
    bign = a / np.sqrt(1 - e2 * sinmu2)
    dx = dlon * bign * np.cos(lat_rad)
    dy = dlat * rm
    return dx, dy


def compute_gtgraph(dd: npt.NDArray[Any]) -> Any:
    """compute the graph of points complying with the speed limits: i and j are
    adjacent if i can be reached by j within the speed limits"""

    import graph_tool as gt

    n = dd.shape[1]
    # horizon = dd.shape[0]
    g = gt.Graph(g=n, directed=True)  # ,g=dd.shape[0])
    eprop_dist = g.new_edge_property("int")
    d = {i: v for i, v in enumerate(g.vertices())}
    edges = [(i, i + h) for h, i in zip(*np.nonzero(dd)) if h > 0]
    g.add_edge_list(edges)  # ,eprops=eprop_dist)
    eprop_dist = g.new_edge_property("int", val=-1)
    return d, g, eprop_dist


def get_gtlongest(dd: npt.NDArray[Any]) -> Any:
    """compute the longest path of points complying with the speed limits"""
    # import graph_tool as gt
    import graph_tool as gt

    v, g, prop_dist = compute_gtgraph(dd)
    # n = g.num_vertices()
    vend = g.add_vertex()
    vstart = g.add_vertex()
    for v in g.iter_vertices():
        if v != vend:
            e = g.add_edge(v, vend)
            prop_dist[e] = -1
        if v != vstart:
            e = g.add_edge(vstart, v)
            prop_dist[e] = -1
    longest_path, _ = gt.topology.shortest_path(
        g,
        source=vstart,
        target=vend,
        weights=prop_dist,
        negative_weights=True,
        dag=True,
    )
    longest_path = list(map(int, longest_path))
    return longest_path


def get_nxlongest(dd: npt.NDArray[Any]) -> Any:
    import networkx as nx

    n = dd.shape[1]
    g = nx.DiGraph()
    for i in range(n):
        g.add_node(i)
    edges = [
        (i, i + h, {"weight": -1}) for h, i in zip(*np.nonzero(dd)) if h > 0
    ]
    g.add_edges_from(edges)
    for i in range(n + 1):
        g.add_edge(-1, i, weight=-1)
    for i in range(-1, n):
        g.add_edge(i, n, weight=-1)
    path = nx.shortest_path(
        g, source=-1, target=n, weight="weight", method="bellman-ford"
    )
    return path


def exact_solver(dd: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    try:
        longest_path = get_gtlongest(dd)  # ,weights)
    except ImportError:
        warnings.warn(
            "graph-tool library not installed, "
            "switching to slower NetworkX library"
        )
        longest_path = get_nxlongest(dd)
    res = np.full(dd.shape[1], True)
    res[np.array(longest_path)[1:-1]] = False
    return res


def approx_solver(dd: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    ddbwd = np.empty_like(dd)
    ddbwd.fill(False)
    for h in range(dd.shape[0]):
        assert dd[h, : dd.shape[1] - h].shape == ddbwd[h, h:].shape
        ddbwd[h, h:] = dd[h, : dd.shape[1] - h]
    res = np.full(dd.shape[1], True)
    out = 10 * res.shape[0]  # 30000

    def selectpoints(iinit, dd, opadd, argmaxopadd):  # type: ignore
        jumpmin = dd[1:].argmax(axis=0) + 1
        jumpmin[(~dd[1:,]).all(axis=0)] = out
        ### computes heuristic
        succeed = np.zeros(res.shape, dtype=np.int64)
        nsteps = 10
        posjumpmin = np.arange(res.shape[0])

        def check_in(posjumpmin: Any) -> npt.NDArray[np.bool_]:
            return np.logical_and(  # type: ignore
                0 <= posjumpmin,
                posjumpmin < res.shape[0],
            )

        for k in range(nsteps):
            valid = check_in(posjumpmin)
            posjumpminvalid = posjumpmin[valid]
            posjumpmin[valid] = opadd(posjumpminvalid, jumpmin[posjumpminvalid])
            succeed[check_in(posjumpmin)] += 1
        posjumpmin[succeed != nsteps] = opadd(
            0, out - succeed[succeed != nsteps]
        )

        #### end compute heuristic
        def selectjump(
            i: int,
            cjump: npt.NDArray[np.int64],
            prediction: npt.NDArray[np.int64],
        ) -> int:
            return cjump[argmaxopadd(prediction[opadd(i, cjump)])]  # type: ignore

        i = iinit
        while 0 <= i < res.shape[0]:
            res[i] = False
            candidatejump = np.arange(1, dd.shape[0])[dd[1:, i]]
            if len(candidatejump) == 0:
                break
            else:
                i = opadd(i, selectjump(i, candidatejump, posjumpmin))

    ss = np.sum(dd[1:], axis=0) + np.sum(ddbwd[1:], axis=0)
    iinit = ss.argmax()
    selectpoints(iinit, dd, operator.add, np.argmin)  # type: ignore
    selectpoints(iinit, ddbwd, operator.sub, np.argmax)  # type: ignore
    return res


def consistency_solver(
    dd: npt.NDArray[Any], exact_when_kept_below: float
) -> npt.NDArray[np.bool_]:
    if exact_when_kept_below == 1:
        return exact_solver(dd)
    else:
        mask = approx_solver(dd)
        if 1 - np.mean(mask) < exact_when_kept_below:
            return exact_solver(dd)
        else:
            return mask


class FilterConsistency(FilterBase):
    """
    Filters noisy values, keeping only values consistent with each other.
    Consistencies are checked between points :math:`i` and points :math:`j \\in
    [|i+1;i+horizon|]`. Using these consistencies, a graph is built: if
    :math:`i` and :math:`j` are consistent, an edge :math:`(i,j)` is added to
    the graph. The kept values is the longest path in this graph, resulting in a
    sequence of consistent values.  The consistencies checked vertically between
    :math:`t_i<t_j` are:: :math:`|(alt_j-alt_i)-(t_j-t_i)* ROCD_i| <
    clip((t_j-t_i)*dalt_dt,dalt_min,dalt_max)` and
    :math:`|(alt_j-alt_i)-(t_j-t_i)* ROCD_j| <
    clip((t_j-t_i)*dalt_dt,dalt_min,dalt_max)` where :math:`dalt_dt`,
    :math:`dalt_min` and :math:`dalt_max` are thresholds that can be specified
    by the user.

    The consistencies checked horizontally between :math:`t_i<t_j` are:
    :math:`|track_i-atan2(lat_j-lat_i,lon_j-lon_i)| < (t_j-t_i)*dtrack_dt` and
    :math:`|track_j-atan2(lat_j-lat_i,lon_j-lon_i)| < (t_j-t_i)*dtrack_dt` and
    :math:`|dist(lat_j,lat_i,lon_j,lon_i)-groundspeed_i*(t_j-t_i)| <
    clip((t_j-t_i)*groundspeed_i*ddist_dt_over_gspeed,0,ddist_max)` where
    :math:`dtrack_dt`, :math:`ddist_dt_over_gspeed` and :math:`ddist_max`
    are thresholds that can be specified by the user.

    If :math:`horiz_and_verti_together=False` then two graphs are built, one
    for the vertical variables and one for horizontal variables. Otherwise, only
    one graph is built. Using two graphs is more precise but slower.  In order
    to compute the longest path faster, a greedy algorithm is used. However, if
    the ratio of kept points is inferior to :math:`exact_when_kept_below`
    then an exact and slower computation is triggered. This computation uses the
    Network library or the faster graph-tool library if available.

    This filter replaces unacceptable values with NaNs. Then, a strategy may be
    applied to fill the NaN values, by default a forward/backward fill. Other
    strategies may be passed, for instance do nothing: None; or interpolate:
    lambda x: x.interpolate(limit_area='inside')

    """

    default: ClassVar[dict[str, float | tuple[float, ...]]] = dict(
        dtrack_dt=2,  # [degree/s]
        dalt_dt=200,  # [feet/min]
        dalt_min=50,  # [feet]
        dalt_max=4000,  # [feet]
        ddist_dt_over_gspeed=0.2,  # [-]
        ddist_max=3,  # [NM]
    )

    def __init__(
        self,
        horizon: int | None = 200,
        horiz_and_verti_together: bool = False,
        exact_when_kept_below: float = 0.5,
        **kwargs: float | tuple[float, ...],
    ) -> None:
        self.horiz_and_verti_together = horiz_and_verti_together
        self.horizon = horizon
        self.thresholds = {**self.default, **kwargs}
        self.exact_when_kept_below = exact_when_kept_below

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        lat = data.latitude.values
        lon = data.longitude.values
        alt = data.altitude.values
        rocd = data.vertical_rate.values
        gspeed = data.groundspeed.values
        t = (
            (data.timestamp - data.timestamp.iloc[0])
            / pd.to_timedelta(1, unit="s")
        ).values
        assert t.min() >= 0
        n = lat.shape[0]
        horizon = n if self.horizon is None else min(self.horizon, n)

        def lagh(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return lag(horizon, v)

        dt = abs(lagh(t) - t) / 3600
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lag_lat_rad = lagh(lat_rad)
        # mask_nan = np.isnan(lag_lat_rad)
        dlat = lag_lat_rad - lat_rad
        dlon = lagh(lon_rad) - lon_rad
        dx, dy = dxdy_from_dlat_dlon(lat_rad, lon_rad, dlat, dlon)
        track_rad = np.radians(data.track.values)
        lag_track_rad = lagh(track_rad)
        mytrack_rad = np.arctan2(-dx, -dy) + np.pi
        thresh_track = np.radians(self.thresholds["dtrack_dt"]) * 3600 * dt
        ddtrack_fwd = np.abs(diffangle(mytrack_rad, track_rad)) < thresh_track
        ddtrack_bwd = (
            np.abs(diffangle(mytrack_rad, lag_track_rad)) < thresh_track
        )
        ddtrack = np.logical_and(ddtrack_fwd, ddtrack_bwd)
        dist = distance(lag_lat_rad, lagh(lon_rad), lat_rad, lon_rad)
        absdist = abs(dist)
        gdist_matrix = gspeed * dt
        ddspeed = np.abs(gdist_matrix - absdist) < np.clip(
            self.thresholds["ddist_dt_over_gspeed"] * gspeed * dt,
            0,
            self.thresholds["ddist_max"],
        )
        ddhorizontal = np.logical_and(ddtrack, ddspeed)

        dalt = lagh(alt) - alt
        thresh_alt = np.clip(
            self.thresholds["dalt_dt"] * 60 * dt,
            self.thresholds["dalt_min"],
            self.thresholds["dalt_max"],
        )
        ddvertical = np.logical_and(
            np.abs(dalt - rocd * dt * 60) < thresh_alt,
            np.abs(dalt - lagh(rocd) * dt * 60) < thresh_alt,
        )
        if self.horiz_and_verti_together:
            mask_horiz = consistency_solver(
                np.logical_and(ddhorizontal, ddvertical),
                self.exact_when_kept_below,
            )
            mask_verti = mask_horiz
        else:
            mask_horiz = consistency_solver(
                ddhorizontal, self.exact_when_kept_below
            )
            mask_verti = consistency_solver(
                ddvertical, self.exact_when_kept_below
            )
        data.loc[
            mask_horiz, ["track", "longitude", "latitude", "groundspeed"]
        ] = np.nan
        data.loc[mask_verti, ["altitude", "vertical_rate"]] = np.nan
        return data
