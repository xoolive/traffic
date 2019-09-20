import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Union
from xml.etree import ElementTree

from requests import Session
from requests_pkcs12 import Pkcs12Adapter
from tqdm.autonotebook import tqdm


class NMB2B:

    PREOPS = {
        "base_url": "https://www.b2b.preops.nm.eurocontrol.int/",
        "post_url": (
            "https://www.b2b.preops.nm.eurocontrol.int/"
            "B2B_PREOPS/gateway/spec/"
        ),
        "file_url": (
            "https://www.b2b.preops.nm.eurocontrol.int/"
            "FILE_PREOPS/gateway/spec/"
        ),
    }

    OPS = {
        "base_url": "https://www.b2b.nm.eurocontrol.int/",
        "post_url": "https://www.b2b.nm.eurocontrol.int/B2B_OPS/gateway/spec/",
        "file_url": "https://www.b2b.nm.eurocontrol.int/FILE_OPS/gateway/spec/",
    }

    REQUESTS = {
        "CompleteAIXMDatasetRequest": """<?xml version="1.0" encoding="UTF-8"?>
<airspace:CompleteAIXMDatasetRequest
    xmlns:airspace="eurocontrol/cfmu/b2b/AirspaceServices">
    <sendTime>{send_time:%Y-%m-%d %H:%M:%S}</sendTime>
    <queryCriteria>
    <airac>
        <airacId>{airac_id}</airacId>
    </airac>
  </queryCriteria>
</airspace:CompleteAIXMDatasetRequest>
"""
    }

    def __init__(
        self,
        mode: Dict[str, str],
        version: str,
        pkcs12_filename: Union[str, Path],
        pkcs12_password: str,
    ) -> None:
        self.mode = mode
        self.version = version
        self.session = Session()
        self.session.mount(
            mode["base_url"],
            Pkcs12Adapter(
                pkcs12_filename=pkcs12_filename, pkcs12_password=pkcs12_password
            ),
        )
        res = self.session.get(mode["base_url"])
        res.raise_for_status()

    def post(self, data: str) -> ElementTree.Element:
        res = self.session.post(
            self.mode["post_url"] + self.version,
            data=data.encode(),
            headers={"Content-Type": "application/xml"},
        )
        res.raise_for_status()
        tree = ElementTree.fromstring(res.content)

        if tree is None:
            raise RuntimeError("Unexpected error")

        if tree.find("status").text != "OK":  # type: ignore
            raise RuntimeError(ElementTree.tostring(tree).decode())

        return tree

    def get(
        self, path: str, output_dir: Union[Path, str] = Path("~").expanduser()
    ) -> None:
        res = self.session.get(self.mode["file_url"] + path, stream=True)
        res.raise_for_status()

        total = int(res.headers["Content-Length"])
        buffer = io.BytesIO()
        for chunk in tqdm(
            res.iter_content(1024),
            total=total // 1024 + 1 if total % 1024 > 0 else 0,
            desc=path.split("/")[-1],
        ):
            buffer.write(chunk)

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir = output_dir.expanduser()

        if not output_dir.exists():
            output_dir.mkdir()

        with (output_dir / path.split("/")[-1]).open("wb") as fh:
            buffer.seek(0)
            fh.write(buffer.read())

    def aixm_dataset(
        self,
        airac_id: Union[str, int],
        output_dir: Union[None, Path, str] = None,
    ):
        data = self.REQUESTS["CompleteAIXMDatasetRequest"].format(
            airac_id=airac_id, send_time=datetime.now()
        )
        tree = self.post(data)

        # There may be several dataset available.
        # For now, we keep the latest one
        latest = max(
            (chunk for chunk in tree.findall("data/datasetSummaries")),
            key=lambda x: x.find("publicationDate").text,  # type: ignore
            default=None,
        )

        if latest is None:
            raise RuntimeError(f"No AIRAC {airac_id} available")

        if output_dir is None:
            output_dir = Path(".") / f"AIRAC_{airac_id}"

        for elt in latest.findall("files"):
            path: str = elt.find("id").text  # type: ignore
            self.get(path, output_dir)
