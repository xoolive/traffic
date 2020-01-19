import io
from hashlib import md5
from typing import Any, Dict

from tqdm.autonotebook import tqdm

from ... import cache_dir
from ...core import Traffic

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
    landing_zurich_2019=dict(
        url="https://ndownloader.figshare.com/files/20291079",
        md5sum="c5577f450424fa74ca673ed8a168c67f",
        filename="landing_dataset.parquet",
        reader=Traffic.from_file,
    ),
)

__all__ = list(datasets.keys())


def download_data(dataset: Dict[str, str]) -> io.BytesIO:
    from .. import session

    f = session.get(dataset["url"], stream=True)
    total = int(f.headers["Content-Length"])
    buffer = io.BytesIO()
    for chunk in tqdm(
        f.iter_content(1024),
        total=total // 1024 + 1 if total % 1024 > 0 else 0,
        desc="download",
    ):
        buffer.write(chunk)

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
