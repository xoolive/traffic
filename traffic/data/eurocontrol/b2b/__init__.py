import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Union
from xml.dom import minidom
from xml.etree import ElementTree

from requests import Session
from tqdm.autonotebook import tqdm

from .flight import FlightManagement
from .flow import Measures
from .pkcs12 import Pkcs12Adapter
from .reply import B2BReply
from .xml import REQUESTS


class NMB2B(FlightManagement, Measures):
    """
    The main instance of this class is provided as:

    .. code:: python

        from traffic.data import nm_b2b

    A path to your certificate and your password must be set in the
    configuration file.

    .. code:: python

        >>> import traffic
        >>> traffic.config_file
        PosixPath('/home/xo/.config/traffic/traffic.conf')

    Then edit the following line accordingly:

    ::

        [nmb2b]
        pkcs12_filename =
        pkcs12_password =

    """

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

    def __init__(
        self,
        mode: Dict[str, str],
        version: str,
        session: Session,
        pkcs12_filename: Union[str, Path],
        pkcs12_password: str,
    ) -> None:
        self.mode = mode
        self.version = version
        self.session = session
        self.session.mount(
            mode["base_url"],
            Pkcs12Adapter(
                filename=Path(pkcs12_filename), password=pkcs12_password
            ),
        )

    def post(self, data: str) -> B2BReply:
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
            rough_string = ElementTree.tostring(tree)
            reparsed = minidom.parseString(rough_string)
            raise RuntimeError(reparsed.toprettyxml(indent="  "))

        return B2BReply.fromET(tree)

    def get(self, path: str, output_dir: Union[None, Path, str] = None) -> None:

        if output_dir is None:
            output_dir = Path("~").expanduser()

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
        data = REQUESTS["CompleteAIXMDatasetRequest"].format(
            airac_id=airac_id, send_time=datetime.now(timezone.utc)
        )
        res = self.post(data)
        assert res.reply is not None

        # There may be several dataset available.
        # For now, we keep the latest one
        latest = max(  # type: ignore
            res.reply.findall("data/datasetSummaries"),
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
