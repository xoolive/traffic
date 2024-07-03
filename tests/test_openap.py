from traffic.data.samples import fuelflow_a320  # type: ignore


def test_openap() -> None:
    f = fuelflow_a320.assign(
        # the vertical_rate is not present in the data
        vertical_rate=lambda df: df.altitude.diff().fillna(0) * 60,
        # convert to kg/s
        fuelflow=lambda df: df.fuelflow / 3600,
    )

    resampled = f.resample("5s")
    openap = resampled.fuelflow(typecode="A320")
    assert openap is not None
    reference = resampled.data.fuelflow
    prediction = openap.data.fuelflow

    assert (reference - prediction).abs().mean() / reference.max() < 0.03
