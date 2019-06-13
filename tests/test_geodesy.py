import numpy as np

from traffic.core.geodesy import bearing, destination, distance, greatcircle


def test_geodesy():

    assert (
        distance(0, 0, 0, 1 / 60) - 1852
    ) / 1852 < 1e-2  # original definition of the nautical mile
    assert destination(0, 0, 0, 123456)[1] == 0
    assert bearing(0, 0, 0, 45) == 90.0
    x = np.stack(greatcircle(0, 0, 0, 45, 44))
    assert sum(x[:, 0]) == 0
    assert sum(x[:, 1]) == 45 * 22

    # Vector version
    d = distance(x[1:, 0], x[1:, 1], x[:-1, 0], x[:-1, 1])
    assert (d.max() - d.min()) / d.max() < 1e-6
    b = bearing(x[:-1, 0], x[:-1, 1], x[1:, 0], x[1:, 1])
    assert (b.max() - b.min()) / b.max() < 1e-6
