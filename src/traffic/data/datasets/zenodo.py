from __future__ import annotations

import hashlib
import json
from operator import itemgetter
from pathlib import Path
from typing import TypedDict

import httpx
from tqdm.rich import tqdm

from ... import cache_dir

client = httpx.Client(follow_redirects=True)


class JSON(TypedDict): ...


class Link(TypedDict):
    self: str


class File(TypedDict):
    id: str
    key: str
    size: int
    checksum: str
    links: Link


class Entry(TypedDict):
    # there are more keys, but we don't use them here
    created: str
    modified: str
    id: int
    doi: str
    metadata: JSON
    files: list[File]


class Hits(TypedDict):
    hits: list[Entry]


class ZenodoSpec(TypedDict):
    hits: Hits


class Zenodo:
    BASE_URL = "https://zenodo.org/api/records/{ident}/versions"

    def __init__(self, ident: str) -> None:
        cache = cache_dir / "datasets" / "zenodo"
        if not cache.exists():
            cache.mkdir(parents=True)
        if not (spec_file := (cache / ident).with_suffix(".json")).exists():
            request = httpx.Request(
                "GET",
                self.BASE_URL.format(ident=ident),
                params={"size": 5, "sort": "version", "allversions": True},
            )
            response = client.send(request)
            response.raise_for_status()
            spec_file.write_text(json.dumps(response.json(), indent=2))

        self.ident = ident
        self.cache = cache
        self.spec: ZenodoSpec = json.loads(spec_file.read_text())

    def get_data(self, key: str) -> Path:
        if not (dirname := self.cache / self.ident).exists():
            dirname.mkdir(parents=True)

        if not (filename := dirname / key).exists():
            version = max(self.spec["hits"]["hits"], key=itemgetter("created"))
            entry = next(e for e in version["files"] if e["key"] == key)

            with filename.open("wb") as file_handle:
                with client.stream("GET", entry["links"]["self"]) as response:
                    md5_hash = hashlib.md5()
                    with tqdm(
                        total=entry["size"],
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                    ) as progress:
                        n_bytes = response.num_bytes_downloaded
                        for chunk in response.iter_bytes():
                            file_handle.write(chunk)
                            md5_hash.update(chunk)
                            progress.update(
                                response.num_bytes_downloaded - n_bytes
                            )
                            n_bytes = response.num_bytes_downloaded

            if (digest := f"md5:{md5_hash.hexdigest()}") != entry["checksum"]:
                filename.unlink()
                msg = (
                    "Mismatch in MD5 hash expected:"
                    f"{entry['checksum']} got: {digest}"
                )
                raise ValueError(msg)

        return filename
