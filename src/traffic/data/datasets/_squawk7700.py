import pandas as pd

from ...core import Traffic
from .zenodo import Zenodo


class Squawk7700Dataset:
    def __init__(self) -> None:
        zenodo = Zenodo("3937483")
        # TODO error in progressbar
        metadata = zenodo.get_data("squawk7700_metadata.csv")
        trajectories = zenodo.get_data("squawk7700_trajectories.parquet.gz")
        self.metadata = pd.read_csv(metadata)

        self.traffic = Traffic.from_file(trajectories)
        assert self.traffic is not None
        self.traffic = self.traffic.merge(
            self.metadata.drop(columns=["callsign", "icao24"]), on="flight_id"
        )
        # TODO remove this (see doc/gallery)
        self.traffic.metadata = self.metadata  # type: ignore
