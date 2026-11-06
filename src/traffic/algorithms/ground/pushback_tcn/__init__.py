from functools import lru_cache
from os import getpid
from pkgutil import get_data
from typing import Any, cast

import numpy as np
import pandas as pd

from ....core.flight import Flight
from ....core.structure import Airport

# The values below are closely intricated with the provided model.
GS_THRESHOLD = 0.5  # in knots, track changes are ignored at lower speeds
TRACK_WINDOWS = (5, 15, 30)  # in samples
ALTITUDE_MARGIN = 500  # in feet, above the airport elevation
SAMPLE_INTERVAL = 1.0  # in seconds


@lru_cache(maxsize=None)
def _load_session(pid: int) -> Any:
    """Loads one inference session per process.

    A session is not shared with forked processes, hence the ``pid`` argument.
    The trajectories of a Traffic are processed one Flight at a time, so
    reusing the session avoids loading the model for each of them.
    """
    import onnxruntime as rt

    pkg = "traffic.algorithms.ground.pushback_tcn"
    model = get_data(pkg, "pushback_tcn.onnx")
    if model is None:
        raise RuntimeError("Missing pushback_tcn.onnx resource")

    # Parallelism is left to Traffic.eval(max_workers=...)
    options = rt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return rt.InferenceSession(
        model, options, providers=["CPUExecutionProvider"]
    )


def _track_change(gs: np.ndarray, track: np.ndarray, n: int) -> np.ndarray:
    """Absolute difference between the mean track of the previous ``n`` and
    the next ``n`` samples, taken over the samples where the aircraft moves."""
    moving = np.flatnonzero(gs > GS_THRESHOLD)
    if len(moving) < 2:
        return np.zeros(len(gs), dtype=np.float32)

    radians = np.radians(track[moving])
    sin_sum = np.concatenate([[0.0], np.cumsum(np.sin(radians))])
    cos_sum = np.concatenate([[0.0], np.cumsum(np.cos(radians))])

    position = np.searchsorted(moving, np.arange(len(gs)))
    before = np.maximum(0, position - n)
    after = np.minimum(len(moving), position + n)
    valid = (position > before) & (after > position)

    def direction(start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        count = np.maximum(1, stop - start)
        sin_mean = (sin_sum[stop] - sin_sum[start]) / count
        cos_mean = (cos_sum[stop] - cos_sum[start]) / count
        angle: np.ndarray = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
        return angle

    diff = direction(position, after) - direction(before, position)
    result: np.ndarray = np.abs((diff + 180) % 360 - 180).astype(np.float32)
    result[~valid] = 0
    return result


def _features(gs: np.ndarray, track: np.ndarray) -> np.ndarray:
    """Computes the input of the model, in the order expected by the model."""
    gs_diff = np.diff(gs, prepend=gs[0]).astype(np.float32)
    gs_diff[gs < GS_THRESHOLD] = 0

    previous_track = np.roll(track, 1)
    track_diff = ((track - previous_track + 180) % 360) - 180
    track_diff = np.abs(track_diff).astype(np.float32)
    track_diff[0] = 0
    track_diff[gs < GS_THRESHOLD] = 0

    return np.column_stack(
        [gs, gs_diff, track_diff]
        + [_track_change(gs, track, n) for n in TRACK_WINDOWS]
    ).astype(np.float32)


def _split(
    seconds: np.ndarray, indices: np.ndarray, max_interruption: float
) -> list[np.ndarray]:
    """Splits indices wherever the timestamps are too far apart."""
    interruption = np.diff(seconds[indices]) - SAMPLE_INTERVAL
    breaks = np.flatnonzero(interruption > max_interruption) + 1
    return np.split(indices, breaks)


class TCNBasedPushback:
    """Returns the pushback part of the trajectory on ground.

    This approach is based on a temporal convolutional network (TCN) which
    estimates a pushback probability for each sample of the trajectory. The
    inputs of the model are computed from the timestamped positions of the
    aircraft: no information about parking positions or apron areas is needed.

    The model has been trained on departures at Zurich (LSZH), Stockholm
    Arlanda (ESSA), Tokyo Haneda (RJTT) and Perth (YPPH), with automatically
    generated labels, from trajectories resampled to 1 second. It has been
    evaluated on manually labelled departures at Brussels (EBBR), Boston (KBOS)
    and Buenos Aires Aeroparque (SABE). Details will be published in a coming
    academic publication. The test dataset is available at
    https://doi.org/10.5281/zenodo.23548217.

    .. warning::

        The model has only been trained and evaluated on departures. It may
        return wrong segments for arrivals, which should be filtered out first.

    At most one segment is returned for each trajectory. The trajectory is
    first cut at the last point on the airport with a ``geoaltitude`` below
    500 ft above the airport elevation or unavailable, then resampled to 1
    second. Without a ``geoaltitude`` feature, it is cut at the last point on
    the airport.

    Samples at or above ``core_threshold`` are grouped into candidates, as long
    as they are not interrupted for more than ``core_max_interruption``. The
    candidate with the best mean probability is then extended to the
    surrounding samples at or above ``edge_threshold``, as long as they are not
    interrupted for more than ``edge_max_interruption``.

    The default core parameters have been selected on the development data;
    the default edge parameters have been set a priori.

    :param airport: Airport where the pushback takes place
    :param core_threshold: the minimum probability of a sample in a candidate
    :param core_max_interruption: the longest interruption within a candidate
    :param edge_threshold: the minimum probability of a sample added around the
      selected candidate; it cannot exceed ``core_threshold``
    :param edge_max_interruption: the longest interruption when extending the
      selected candidate; it cannot be shorter than ``core_max_interruption``

    >>> from traffic.data.samples import zurich_airport
    >>> flight = zurich_airport["AEE5ZH"]
    >>> pushback = flight.pushback("LSZH", method="tcn")
    >>> pushback.duration
    Timedelta('0 days 00:02:27')
    """

    def __init__(
        self,
        airport: str | Airport,
        core_threshold: float = 0.84,
        core_max_interruption: str = "15s",
        edge_threshold: float = 0.5,
        edge_max_interruption: str = "30s",
    ) -> None:
        from ....data import airports

        self.airport = (
            airports[airport] if isinstance(airport, str) else airport
        )
        if self.airport is None:
            raise RuntimeError("Airport information missing")

        self.core_threshold = core_threshold
        self.core_max_interruption = pd.Timedelta(
            core_max_interruption
        ).total_seconds()
        self.edge_threshold = edge_threshold
        self.edge_max_interruption = pd.Timedelta(
            edge_max_interruption
        ).total_seconds()
        if self.edge_threshold > self.core_threshold:
            raise ValueError("edge_threshold cannot exceed core_threshold")
        if self.edge_max_interruption < self.core_max_interruption:
            raise ValueError(
                "edge_max_interruption cannot be shorter than "
                "core_max_interruption"
            )

    def _select_segment(
        self, seconds: np.ndarray, probabilities: np.ndarray
    ) -> None | np.ndarray:
        """Selects the indices of the most likely pushback segment.

        Ties on the mean probability are resolved by selecting the longest
        candidate, then the best maximum probability, then the earliest.
        """
        core = np.flatnonzero(probabilities >= self.core_threshold)
        if core.size == 0:
            return None

        best = max(
            _split(seconds, core, self.core_max_interruption),
            key=lambda candidate: (
                float(probabilities[candidate].mean()),
                seconds[candidate[-1]] - seconds[candidate[0]],
                float(probabilities[candidate].max()),
            ),
        )

        # The checks in the constructor ensure that a segment contains the
        # whole candidate.
        edge = np.flatnonzero(probabilities >= self.edge_threshold)
        return next(
            segment
            for segment in _split(seconds, edge, self.edge_max_interruption)
            if segment[0] <= best[0] <= segment[-1]
        )

    def apply(self, flight: Flight) -> None | Flight:
        clipped = flight.clip(self.airport.shape)
        if clipped is None:
            return None

        # Without geometric altitude, only the airport bounds are considered.
        low_points: None | Flight = clipped
        if "geoaltitude" in flight.data:
            limit = self.airport.altitude + ALTITUDE_MARGIN
            low_points = clipped.query(
                f"geoaltitude < {limit} or geoaltitude.isnull()"
            )
        if low_points is None:
            return None

        on_ground = flight.before(low_points.stop, strict=False)
        if on_ground is None or len(on_ground) < 2:
            return None

        # With pandas 3, gaps in pyarrow columns are not interpolated.
        valid = (
            on_ground.assign(
                latitude=on_ground.data.latitude.astype("float64"),
                longitude=on_ground.data.longitude.astype("float64"),
            )
            .resample("1s", how={"interpolate": ["latitude", "longitude"]})
            .cumulative_distance()
            .query("compute_gs.notnull() and compute_track.notnull()")
        )
        if valid is None:
            return None

        data = valid.data
        features = _features(
            data.compute_gs.to_numpy(dtype=np.float32),
            data.compute_track.to_numpy(dtype=np.float32),
        )

        session = _load_session(getpid())
        probabilities = cast(
            np.ndarray, session.run(None, {"features": features[np.newaxis]})
        )[0][0]

        timestamps = data.timestamp.reset_index(drop=True)
        seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
        segment = self._select_segment(seconds.to_numpy(), probabilities)
        if segment is None:
            return None

        return flight.between(
            timestamps.iloc[segment[0]],
            timestamps.iloc[segment[-1]],
            strict=False,
        )
