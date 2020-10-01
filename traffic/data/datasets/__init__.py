import io
from hashlib import md5
from typing import Any, Dict

import pandas as pd
from tqdm.autonotebook import tqdm

from ... import cache_dir
from ...core import Traffic

_squawk7700_url = "https://zenodo.org/record/3937483/"


def _squawk7700_reader(filename):
    metadata = get_dataset(
        dict(
            url=f"{_squawk7700_url}/files/squawk7700_metadata.csv",
            md5sum="4d33ff84a05742136f3733692bf944e6",
            filename="squawk7700_metadata.csv",
            reader=pd.read_csv,
        )
    )
    squawk7700 = Traffic.from_file(filename)
    assert squawk7700 is not None
    squawk7700 = squawk7700.merge(
        metadata.drop(columns=["callsign", "icao24"]), on="flight_id"
    )
    squawk7700.metadata = metadata  # type: ignore
    return squawk7700


datasets = dict(
    paris_toulouse_2017=dict(
        url="https://ndownloader.figshare.com/files/20291055",
        md5sum="e869a60107fdb9f092f5abbb3b43a2c0",
        filename="city_pair_dataset.parquet",
        reader=Traffic.from_file,
    ),
    airspace_bordeaux_2017=dict(
        url="https://ndownloader.figshare.com/files/20291040",
        md5sum="f7b057f12cc735a15b93984b9ae7b8fc",
        filename="airspace_dataset.parquet",
        reader=Traffic.from_file,
    ),
    landing_toulouse_2017=dict(
        url="https://ndownloader.figshare.com/files/24926849",
        md5sum="141e6c39211c382e5dd8ec66096b3798",
        filename="toulouse2017.parquet.gz",
        reader=Traffic.from_file,
    ),
    landing_zurich_2019=dict(
        url="https://ndownloader.figshare.com/files/20291079",
        md5sum="c5577f450424fa74ca673ed8a168c67f",
        filename="landing_dataset.parquet",
        reader=Traffic.from_file,
    ),
    squawk7700=dict(
        url=f"{_squawk7700_url}/files/squawk7700_trajectories.parquet.gz",
        md5sum="8ed5375d92bd7b7b94ea03c0533d959e",
        filename="squawk7700_trajectories.parquet.gz",
        reader=_squawk7700_reader,
    ),
)

__all__ = list(datasets.keys())


def download_data(dataset: Dict[str, str]) -> io.BytesIO:
    from .. import session

    f = session.get(dataset["url"], stream=True)
    buffer = io.BytesIO()

    if "Content-Length" in f.headers:
        total = int(f.headers["Content-Length"])
        for chunk in tqdm(
            f.iter_content(1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc="download",
        ):
            buffer.write(chunk)
    else:
        buffer.write(f.content)

    buffer.seek(0)

    compute_md5 = md5(buffer.getbuffer()).hexdigest()
    if compute_md5 != dataset["md5sum"]:
        raise RuntimeError(
            f"Error in MD5 check: {compute_md5} instead of {dataset['md5sum']}"
        )

    return buffer


def get_dataset(dataset: Dict[str, Any]) -> Any:
    dataset_dir = cache_dir / "datasets"

    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True)

    filename = dataset_dir / dataset["filename"]
    if not filename.exists():
        buffer = download_data(dataset)
        with filename.open("wb") as fh:
            fh.write(buffer.getbuffer())

    return dataset["reader"](filename)


def __getattr__(name: str) -> Any:
    if name not in datasets:
        raise AttributeError(f"No such dataset: {name}")
    return get_dataset(datasets[name])
