from pathlib import Path
from pkgutil import get_data
from typing import Iterator

import numpy as np
import pandas as pd

from ....core.flight import Flight


class MLHoldingDetection:
    """Iterates on all holding pattern segments in the trajectory.

    This approach is based on a neuronal network model. Details will be
    published in a coming academic publication.

    Parameters should be left as default as they are strongly coupled with
    the proposed model.

    The model has been trained on manually labelled holding patterns for
    trajectories landing at different European airports including London
    Heathrow.

    All the parameters below are closely intricated with a model. Default
    parameters should not be changed if we stick with the provided model.

    :param duration: the duration of each sliding window
    :param step: the step for each sliding window
    :param threshold: the minimum duration for each sliding window
    :param samples: each sliding window is resampled to a fixed number of points
    :param model_path: where the models are located, by default in the same
      module directory
    :param vertical_rate: a boolean set to True if the vertical rate should be
      taken into account in the model.

    >>> from traffic.data.samples import belevingsvlucht
    >>> belevingsvlucht.has("holding_pattern")
    True

    See also: :ref:`How to detect holding patterns in aircraft trajectories?`

    (new in version 2.8)
    """

    def __init__(
        self,
        duration: str = "6 min",
        step: str = "2 min",
        threshold: str = "5 min",
        samples: int = 30,
        model_path: None | str | Path = None,
        vertical_rate: bool = False,
    ) -> None:
        import onnxruntime as rt

        self.duration = duration
        self.step = step
        self.threshold = threshold
        self.samples = samples
        self.vertical_rate = vertical_rate

        providers = rt.get_available_providers()

        if model_path is None:
            pkg = "traffic.algorithms.navigation.holding_pattern"
            data = get_data(pkg, "scaler.onnx")
            self.scaler_sess = rt.InferenceSession(data, providers=providers)
            data = get_data(pkg, "classifier.onnx")
            self.classifier_sess = rt.InferenceSession(
                data, providers=providers
            )
        else:
            model_path = Path(model_path)
            self.scaler_sess = rt.InferenceSession(
                (model_path / "scaler.onnx").read_bytes(),
                providers=providers,
            )
            self.classifier_sess = rt.InferenceSession(
                (model_path / "classifier.onnx").read_bytes(),
                providers=providers,
            )

    def apply(self, flight: Flight) -> Iterator[Flight]:
        start, stop = None, None

        for i, window in enumerate(
            flight.sliding_windows(self.duration, self.step)
        ):
            if window.duration >= pd.Timedelta(self.threshold):
                window = window.assign(flight_id=str(i))
                resampled = window.resample(self.samples)

                if resampled.data.eval("track.isnull()").any():
                    continue

                features = (
                    resampled.data.track_unwrapped
                    - resampled.data.track_unwrapped[0]
                ).values.reshape(1, -1)

                if self.vertical_rate:
                    if resampled.data.eval("vertical_rate.notnull()").any():
                        continue
                    vertical_rates = (
                        resampled.data.vertical_rate.values.reshape(1, -1)
                    )
                    features = np.concatenate(
                        (features, vertical_rates), axis=1
                    )

                name = self.scaler_sess.get_inputs()[0].name
                value = features.astype(np.float32)
                x = self.scaler_sess.run(None, {name: value})[0]

                name = self.classifier_sess.get_inputs()[0].name
                value = x.astype(np.float32)
                pred = self.classifier_sess.run(None, {name: value})[0]

                if bool(pred.round().item()):
                    if start is None:
                        start, stop = window.start, window.stop
                    elif start < stop:
                        stop = window.stop
                    else:
                        yield flight.between(start, stop)
                        start, stop = window.start, window.stop

        if start is not None:
            yield flight.between(start, stop)  # type: ignore
