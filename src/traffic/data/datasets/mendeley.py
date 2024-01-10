from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import httpx
from tqdm.rich import tqdm

from ... import cache_dir

client = httpx.Client(follow_redirects=True)


class ContentDetails(TypedDict):
    id: str
    sha256_hash: str
    content_type: str
    size: int
    created_date: str
    download_url: str
    view_url: str
    download_expiry_time: str


class MendeleyEntry(TypedDict):
    filename: str
    id: str
    content_details: ContentDetails
    size: int
    last_modified_date: str
    status: str


class Mendeley:
    BASE_URL = "https://data.mendeley.com/public-api/datasets/{ident}/files"

    def __init__(self, ident: str) -> None:
        cache = cache_dir / "datasets" / "mendeley"
        if not cache.exists():
            cache.mkdir(parents=True)
        if not (spec_file := (cache / ident).with_suffix(".json")).exists():
            request = httpx.Request(
                "GET",
                self.BASE_URL.format(ident=ident),
                params={"folder_id": "root", "version": 1},
            )
            response = client.send(request)
            response.raise_for_status()
            spec_file.write_text(json.dumps(response.json(), indent=2))

        self.ident = ident
        self.cache = cache
        self.spec: list[MendeleyEntry] = json.loads(spec_file.read_text())

    def get_data(self, key: str) -> Path:
        if not (dirname := self.cache / self.ident).exists():
            dirname.mkdir(parents=True)

        if not (filename := dirname / key).exists():
            entry = next(e for e in self.spec if e["filename"] == key)

            with filename.open("wb") as file_handle:
                with client.stream(
                    "GET", entry["content_details"]["download_url"]
                ) as response:
                    total = int(response.headers["Content-Length"])
                    if total != entry["size"]:
                        raise ValueError("Mismatch in expected file size")

                    sha256_hash = hashlib.sha256()

                    with tqdm(
                        total=entry["size"],
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                    ) as progress:
                        n_bytes = response.num_bytes_downloaded
                        for chunk in response.iter_bytes():
                            file_handle.write(chunk)
                            sha256_hash.update(chunk)
                            progress.update(
                                response.num_bytes_downloaded - n_bytes
                            )
                            n_bytes = response.num_bytes_downloaded

            digest = sha256_hash.hexdigest()
            if digest != entry["content_details"]["sha256_hash"]:
                filename.unlink()
                msg = (
                    "Mismatch in SHA256 hash expected:"
                    f"{entry['content_details']['sha256_hash']} got: {digest}"
                )
                raise ValueError(msg)

        return filename
