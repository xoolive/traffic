from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj

if TYPE_CHECKING:
    from cartopy import crs

    from ..core import Traffic


class ScalerProtocol(Protocol):
    def fit_transform(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...

    def inverse_transform(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


class GenerationProtocol(Protocol):
    def fit(
        self, X: npt.NDArray[np.float64], **kwargs: Any
    ) -> "GenerationProtocol": ...

    def sample(
        self, n_samples: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


class Coordinates(TypedDict):
    latitude: float
    longitude: float


def compute_latlon_from_trackgs(
    data: pd.DataFrame,
    n_samples: int,
    n_obs: int,
    coordinates: Coordinates,
    forward: bool = True,
) -> pd.DataFrame:
    """Computes iteratively coordinates (latitude/longitude) using previous
    coordinates, track angle, groundspeed and timedelta.

    The coordinates computation is made using pyproj.Geod.fwd() method.

    data: pandas.DataFrame
        The DataFrame containing trajectories.

    n_samples: int
        Number of trajectories in data.

    n_obs: int
        Number of coordinates representing a trajectory.

    coordinates: Dict[str, float]
        Some coordinates. It should have ``'latitude'`` and ``'longitude'``
        keys.

    forward: bool, default: True
        Whether the coordinates correspond to the first or the last
        coordinates of the trajectories.
    """

    from pitot.geodesy import destination

    df = data.copy(deep=True)
    if not forward:
        df["track"] = df["track"].values[::-1] - 180
        df["groundspeed"] = df["groundspeed"].values[::-1]
        df["timestamp"] = df["timestamp"].values[::-1]
    lat = np.array(
        ([coordinates["latitude"]] + [np.nan] * (n_obs - 1)) * n_samples
    )
    lon = np.array(
        ([coordinates["longitude"]] + [np.nan] * (n_obs - 1)) * n_samples
    )

    for i in range(len(df)):
        if np.isnan(lat[i]) or np.isnan(lon[i]):
            lon1 = lon[i - 1]
            lat1 = lat[i - 1]
            track = df.loc[i - 1, "track"]
            gs = df.loc[i - 1, "groundspeed"]
            delta_time = abs(
                (
                    df.loc[i, "timestamp"] - df.loc[i - 1, "timestamp"]
                ).total_seconds()
            )
            coeff = 0.99
            d = coeff * gs * delta_time * (1852 / 3600)
            lat2, lon2, _ = destination(lat1, lon1, track, d)
            lat[i] = lat2
            lon[i] = lon2

    if not forward:
        lat, lon = lat[::-1], lon[::-1]

    return data.assign(latitude=lat, longitude=lon)


class Generation:
    """Generation class to handle trajectory generation.

    generation: GenerationProtocol
        generation model, should implement ``fit()`` and ``sample()``
        methods.

    features: List[str]
        List of features to generate. Example:
        ``['latitude', 'longitude', 'altitude', 'timedelta']``.

    scaler: ScalerProtocol, default: None
        *if need be*, apply a scaler to the data before fitting the
        generation model. You may want to consider `StandardScaler()
        <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
        The scaler object should implement ``fit_transform()`` and
        ``inverse_transform()`` methods.
    """

    _required_traffic_columns = (
        "altitude",
        "callsign",
        "icao24",
        "latitude",
        "longitude",
        "timestamp",
    )

    _repr_indent = 4

    def __init__(
        self,
        generation: GenerationProtocol,
        features: List[str],
        scaler: Optional[ScalerProtocol] = None,
    ) -> None:
        self.generation = generation
        self.features = features
        self.scaler = scaler

    def prepare_features(self, t: "Traffic") -> npt.NDArray[np.float64]:
        X = np.stack(list(f.data[self.features].values.ravel() for f in t))
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        return X  # type: ignore

    def build_traffic(
        self,
        X: npt.NDArray[np.float64],
        projection: Union[pyproj.Proj, "crs.Projection", None] = None,
        coordinates: Optional[Coordinates] = None,
        forward: bool = True,
    ) -> "Traffic":
        """Build Traffic DataFrame from numpy array according to the list
        of features ``self.features``.
        """
        from ..core import Traffic

        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1, len(self.features))
        n_obs = X.shape[1]
        df = pd.DataFrame(
            {
                feature: X[:, :, i].ravel()
                for i, feature in enumerate(self.features)
            }
        )
        # enriches DataFrame with flight_id, callsign and icao24 columns.
        ids = np.array(
            [[f"TRAJ_{sample}"] * n_obs for sample in range(n_samples)]
        ).ravel()
        df = df.assign(flight_id=ids, callsign=ids, icao24=ids)
        # enriches DataFrame with timestamp columns.
        if "timedelta" in df.columns:
            base_ts = pd.Timestamp.today(tz="UTC").round(freq="s")
            df = df.assign(
                timestamp=pd.to_timedelta(df.timedelta, unit="s") + base_ts
            )
        # if relevant, enriches DataFrame with latitude and longitude columns.
        if not set(["latitude", "longitude"]).issubset(set(self.features)):
            if set(["x", "y"]).issubset(self.features):
                return Traffic(df).compute_latlon_from_xy(projection)
                # df = compute_latlon_from_xy(df, projection=projection)
            if set(["track", "groundspeed"]).issubset(set(self.features)):
                assert (
                    coordinates is not None
                ), "coordinates attribute shouldn't be None"
                # integrate lat/lon in df
                df = compute_latlon_from_trackgs(
                    df, n_samples, n_obs, coordinates, forward=forward
                )

        return Traffic(df)

    def fit(self, t: "Traffic", **kwargs: Any) -> "Generation":
        X = self.prepare_features(t)
        self.generation.fit(X)
        return self

    def sample(
        self,
        n_samples: int = 1,
        projection: Union[pyproj.Proj, "crs.Projection", None] = None,
        coordinates: Optional[Coordinates] = None,
        forward: bool = True,
    ) -> "Traffic":
        """Samples trajectories from the generation model.

        n_samples: int, default: 1
            Number of trajectories to sample.

        projection: pyproj.Proj, cartopy.Projection, default: None
            Required if the generation model uses ``x`` and ``y`` projections
            instead of ``latitude`` and ``longitude``.

        coordinates: Dict[str, float], default: None
            Required if the generation model uses ``track`` and ``groundspeed``
            instead of ``latitude`` and ``longitude``. It should have
            ``'latitude'`` and ``'longitude'`` keys. Example:
            ``{'latitude': 12.2, 'longitude': 43.5}``.

        forward: bool, default: True
            Indicates whether the ``coordinates`` attribute corresponds to
            the first coordinate of the trajectories or the last one. If
            ``True`` it is the first, else it is the last.

        Example usage:
            .. code-block:: python

                # Generation of 10 trajectories with track and groundspeed
                # features, considering some ending coordinates for each
                # trajectories.
                t_gen = g.sample(
                    10,
                    coordinates={"latitude": 15, "longitude":15},
                    forward=False,
                )
        """
        X, _ = self.generation.sample(n_samples)
        if self.scaler is not None:
            X = self.scaler.inverse_transform(X)
        return self.build_traffic(X, projection, coordinates, forward)

    @classmethod
    def from_file(self, path: Union[str, Path]) -> "Generation":
        raise NotImplementedError()

    def save(self, path: Union[str, Path]) -> None:
        raise NotImplementedError()

    def __repr__(self) -> str:
        head = "Generation"
        body = [f"Generative model: {self.generation!r}"]
        body += [f"Features: {self.features}"]
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
