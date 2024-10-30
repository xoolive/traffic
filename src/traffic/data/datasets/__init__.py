from __future__ import annotations

from typing import Any

from ... import config
from ...core import Traffic
from ._squawk7700 import Squawk7700Dataset
from .default import Default, Entry

airspace_bordeaux_2017: Traffic
landing_amsterdam_2019: Traffic
landing_cdg_2019: Traffic
landing_dublin_2019: Traffic
landing_heathrow_2019: Traffic
landing_londoncity_2019: Traffic
landing_toulouse_2017: Traffic
landing_zurich_2019: Traffic
paris_toulouse_2017: Traffic

datasets: dict[str, Entry] = dict(
    paris_toulouse_2017=dict(
        url="https://ndownloader.figshare.com/files/20291055",
        md5sum="e869a60107fdb9f092f5abbb3b43a2c0",
        filename="city_pair_dataset.parquet",
    ),
    airspace_bordeaux_2017=dict(
        url="https://ndownloader.figshare.com/files/20291040",
        md5sum="f7b057f12cc735a15b93984b9ae7b8fc",
        filename="airspace_dataset.parquet",
    ),
    landing_toulouse_2017=dict(
        url="https://ndownloader.figshare.com/files/24926849",
        md5sum="141e6c39211c382e5dd8ec66096b3798",
        filename="toulouse2017.parquet.gz",
    ),
    landing_zurich_2019=dict(
        url="https://ndownloader.figshare.com/files/20291079",
        md5sum="c5577f450424fa74ca673ed8a168c67f",
        filename="landing_dataset.parquet",
    ),
    landing_dublin_2019=dict(
        url="https://data.4tu.nl/file/4e042fbc-4f76-4f28-ac4b-a0120558ceba/94ec5814-6ee9-4cbc-88f2-ca5e1e0dfbf8",
        md5sum="73cc3b882df958cc3b5de547740a5006",
        filename="EIDW_dataset.parquet",
    ),
    landing_cdg_2019=dict(
        url="https://data.4tu.nl/file/4e042fbc-4f76-4f28-ac4b-a0120558ceba/0ad60c2d-a63d-446a-976f-d61fe262c144",
        md5sum="9a2af398037fbfb66f16bf171ca7cf93",
        filename="LFPG_dataset.parquet",
    ),
    landing_amsterdam_2019=dict(
        url="https://data.4tu.nl/file/4e042fbc-4f76-4f28-ac4b-a0120558ceba/901d842e-658c-40a6-98cc-64688a560f57",
        md5sum="419ab7390ee0f3deb0d46fbeecc29c57",
        filename="EHAM_dataset.parquet",
    ),
    landing_heathrow_2019=dict(
        url="https://data.4tu.nl/file/4e042fbc-4f76-4f28-ac4b-a0120558ceba/b40a8064-3b90-4416-8a32-842104b21e4d",
        md5sum="161470e4e93f088cead98178408aa8d1",
        filename="EGLL_dataset.parquet",
    ),
    landing_londoncity_2019=dict(
        url="https://data.4tu.nl/file/4e042fbc-4f76-4f28-ac4b-a0120558ceba/74006294-2a66-4b63-9e32-424da2f74201",
        md5sum="a7ff695355759e72703f101a1e43298c",
        filename="EGLC_dataset.parquet",
    ),
)

squawk7700: Traffic

__all__ = [*list(datasets.keys()), "squawk7700"]

if "datasets" in config:
    __all__ = __all__ + list(config["datasets"].keys())


def __getattr__(name: str) -> Any:
    if on_disk := config.get("datasets", name, fallback=None):
        return Traffic.from_file(on_disk)

    if name == "squawk7700":
        return Squawk7700Dataset().traffic

    if name not in datasets:
        raise AttributeError(f"No such dataset: {name}")

    filename = Default().get_data(datasets[name])
    return Traffic.from_file(filename)
