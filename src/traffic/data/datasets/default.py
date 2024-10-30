import hashlib
from pathlib import Path
from typing import TypedDict

import httpx

from ... import cache_dir, tqdm_style
from ...core import tqdm

client = httpx.Client(follow_redirects=True)


class Entry(TypedDict):
    url: str
    md5sum: str
    filename: str


class Default:
    def __init__(self) -> None:
        cache = cache_dir / "datasets" / "default"
        if not cache.exists():
            cache.mkdir(parents=True)

        self.cache = cache

    def get_data(self, entry: Entry) -> Path:
        if not (filename := self.cache / entry["filename"]).exists():
            md5_hash = hashlib.md5()
            with filename.open("wb") as file_handle:
                with client.stream("GET", entry["url"]) as response:
                    content_length = response.headers.get("Content-Length")
                    if content_length is None:
                        content = response.content
                        file_handle.write(content)
                        md5_hash.update(content)
                    else:
                        if tqdm_style == "silent":
                            for chunk in response.iter_bytes():
                                file_handle.write(chunk)
                                md5_hash.update(chunk)
                        else:
                            with tqdm(  # type: ignore
                                total=int(content_length),
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

            if (digest := md5_hash.hexdigest()) != entry["md5sum"]:
                filename.unlink()
                msg = (
                    "Mismatch in MD5 hash expected:"
                    f"{entry['md5sum']} got: {digest}"
                )
                raise ValueError(msg)

        return filename
