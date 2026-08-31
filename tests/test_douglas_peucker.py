import sys

import pytest

import numpy as np
import pandas as pd
from traffic.algorithms.douglas_peucker import douglas_peucker
from traffic.core import Flight


def _recursive_reference_2d(
    x: np.ndarray, y: np.ndarray, tolerance: float
) -> np.ndarray:
    """Independent recursive RDP mask used as ground truth on small inputs."""
    mask = np.ones(len(x), dtype=bool)

    def rec(start: int, end: int) -> None:
        if end - start < 3:
            return
        xs, ys = x[start:end], y[start:end]
        v = np.array([[ys[-1] - ys[0]], [xs[0] - xs[-1]]])
        denom = np.sqrt(np.sum(v * v))
        if denom == 0:
            return
        d = np.abs(
            np.dot(
                np.dstack([xs[1:-1] - xs[0], ys[1:-1] - ys[0]])[0],
                v / denom,
            )
        )
        if len(d) == 0 or np.max(d) < tolerance:
            mask[start + 1 : end - 1] = False
            return
        arg = int(np.argmax(d))
        farthest = start + arg + 1
        rec(start, farthest + 1)
        rec(farthest, end)

    rec(0, len(x))
    return mask


def _recursive_reference_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, tolerance: float
) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)

    def rec(start: int, end: int) -> None:
        if end - start < 3:
            return
        xs, ys, zs = x[start:end], y[start:end], z[start:end]
        start_pt = np.array([xs[0], ys[0], zs[0]])
        end_pt = np.array([xs[-1], ys[-1], zs[-1]])
        norm = np.linalg.norm(start_pt - end_pt)
        if norm == 0:
            return
        point = np.dstack([xs[1:], ys[1:], zs[1:]])[0] - start_pt
        d = np.cross(point, (start_pt - end_pt) / norm)
        d = np.sqrt(np.sum(d * d, axis=1))
        if len(d) == 0 or np.max(d) < tolerance:
            mask[start + 1 : end - 1] = False
            return
        arg = int(np.argmax(d))
        farthest = start + arg + 1
        rec(start, farthest + 1)
        rec(farthest, end)

    rec(0, len(x))
    return mask


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_douglas_peucker_matches_reference_2d(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 40
    x = np.cumsum(rng.uniform(size=n))
    y = np.cumsum(rng.uniform(size=n))
    for tolerance in (0.1, 0.5, 2.0):
        expected = _recursive_reference_2d(x, y, tolerance)
        got = douglas_peucker(
            df=pd.DataFrame({"x": x, "y": y}), tolerance=tolerance, x="x", y="y"
        )
        assert np.array_equal(got, expected)

    # A straight line collapses to its endpoints.
    line = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0) * 2})
    mask = douglas_peucker(df=line, tolerance=1.0, x="x", y="y")
    assert mask.sum() == 2
    assert mask[0] and mask[-1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_douglas_peucker_matches_reference_3d(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 40
    x = np.cumsum(rng.uniform(size=n))
    y = np.cumsum(rng.uniform(size=n))
    z = np.cumsum(rng.uniform(size=n))
    for tolerance in (0.1, 0.5, 2.0):
        expected = _recursive_reference_3d(x, y, z, tolerance)
        # z_factor=1.0 so the implementation scales z identically to the
        # reference (which operates on raw z).
        got = douglas_peucker(
            df=pd.DataFrame({"x": x, "y": y, "z": z}),
            tolerance=tolerance,
            x="x",
            y="y",
            z="z",
            z_factor=1.0,
        )
        assert np.array_equal(got, expected)


def test_douglas_peucker_adversarial_matches_reference() -> None:
    # A period-2 sawtooth is the RDP worst case: each level removes no point
    # yet recurses one level deeper, reaching depth ~n. Verify the iterative
    # result matches the recursive reference on a small such input.
    n = 60
    x = np.arange(n, dtype=float)
    y = (np.arange(n) % 2).astype(float)
    expected = _recursive_reference_2d(x, y, 0.5)
    got = douglas_peucker(
        df=pd.DataFrame({"x": x, "y": y}), tolerance=0.5, x="x", y="y"
    )
    assert np.array_equal(got, expected)


def test_douglas_peucker_adversarial_input_no_recursion() -> None:
    # The same period-2 sawtooth at n=2000 forces depth ~1998, which exceeded
    # Python's recursion limit and raised RecursionError, notably on
    # Python 3.14 (issue #568). The iterative implementation must not depend
    # on Python recursion depth at all.
    n = 2000
    x = np.arange(n, dtype=float)
    y = (np.arange(n) % 2).astype(float)
    df = pd.DataFrame({"x": x, "y": y})

    original = sys.getrecursionlimit()
    sys.setrecursionlimit(500)
    try:
        mask = douglas_peucker(df=df, tolerance=0.5, x="x", y="y")
    finally:
        sys.setrecursionlimit(original)

    assert mask.dtype == bool
    assert len(mask) == n
    assert mask[0] and mask[-1]
    # The sawtooth keeps every point (no interior point is within tolerance
    # of the running chord), so the value of this assertion is that the call
    # returned at all rather than raising RecursionError.
    assert mask.sum() == n


def test_flight_simplify_long_trajectory() -> None:
    n = 2000
    x = np.arange(n, dtype=float)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01", periods=n, freq="1s", tz="UTC"
            ),
            "x": x,
            "y": np.sin(x / 10.0),
            "altitude": np.zeros(n),
        }
    )
    flight = Flight(data)
    simplified = flight.simplify(0.2)
    assert isinstance(simplified, Flight)
    assert 2 <= len(simplified) < n
