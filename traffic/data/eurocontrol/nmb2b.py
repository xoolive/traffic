import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Union
from xml.dom import minidom
from xml.etree import ElementTree

import pandas as pd
from requests import Session
from requests_pkcs12 import Pkcs12Adapter
from tqdm.autonotebook import tqdm

from ...core.mixins import DataFrameMixin

rename_cols = {
    "aircraftId": "callsign",
    "aircraftType": "typecode",
    "aerodromeOfDeparture": "origin",
    "aerodromeOfDestination": "destination",
    "estimatedOffBlockTime": "EOBT",
    "estimatedTakeOffTime": "ETOT",
    "calculatedTakeOffTime": "CTOT",
    "actualTakeOffTime": "ATOT",
    "estimatedTimeOfArrival": "ETOA",
    "calculatedTimeOfArrival": "CTOA",
    "actualTimeOfArrival": "ATOA",
    "aircraftAddress": "icao24",
}


class B2BReply:
    def __init__(self, *args, **kwargs) -> None:
        self.reply: Optional[ElementTree.Element] = None

    @classmethod
    def fromET(cls, tree: ElementTree.Element) -> "B2BReply":
        instance = cls()
        instance.reply = tree
        return instance

    def __str__(self) -> str:
        if self.reply is None:
            return "[empty]"
        s = ElementTree.tostring(self.reply)
        return minidom.parseString(s).toprettyxml(indent="  ")

    def __repr__(self) -> str:
        res = str(self)
        if len(res) > 1000:
            return res[:1000] + "..."
        return res


class FlightList(DataFrameMixin, B2BReply):
    def __init__(self, *args, **kwargs):
        if len(args) == 0 and "data" not in kwargs:
            super().__init__(data=None, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def fromB2BReply(cls, r: B2BReply):
        assert r.reply is not None
        return cls.fromET(r.reply)

    @classmethod
    def fromET(cls, tree: ElementTree.Element) -> "B2BReply":
        instance = cls()
        instance.reply = tree
        instance.build_df()
        return instance

    def __getitem__(self, item) -> "B2BReply":
        from ...data import nmb2b

        data = nmb2b.REQUESTS["FlightRetrievalRequest"].format(
            send_time=datetime.now(timezone.utc), flight_id=item
        )

        return nmb2b.post(data)

    def _ipython_key_completions_(self) -> Set[str]:
        return set(self.data.flightId.unique())

    def build_df(self):

        self.data = pd.DataFrame.from_records(
            [
                {
                    **{"flightId": elt.find("flightId/id").text},
                    **{p.tag: p.text for p in elt.find("flightId/keys")},
                    **{
                        p.tag: p.text
                        for p in elt
                        if p.tag != "flightId" and p.text is not None
                    },
                }
                for elt in self.reply.findall("data/flights/flight")
            ]
        )
        if "nonICAOAerodromeOfDeparture" in self.data.columns:
            self.data = self.data.drop(
                columns=[
                    "nonICAOAerodromeOfDeparture",
                    "nonICAOAerodromeOfDestination",
                    "airFiled",
                ]
            )

        self.data = self.data.rename(columns=rename_cols)

        for feat in ["EOBT", "ETOT", "CTOT", "ATOT", "ETOA", "CTOA", "ATOA"]:
            if feat in self.data.columns:
                self.data = self.data.assign(
                    **{
                        feat: self.data[feat].apply(
                            lambda x: pd.Timestamp(x, tz="utc")
                        )
                    }
                )

        if "icao24" in self.data.columns:
            self.data = self.data.assign(icao24=self.data.icao24.str.lower())


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
        "CompleteAIXMDatasetRequest": """
<airspace:CompleteAIXMDatasetRequest
    xmlns:airspace="eurocontrol/cfmu/b2b/AirspaceServices">
    <sendTime>{send_time:%Y-%m-%d %H:%M:%S}</sendTime>
    <queryCriteria>
    <airac>
        <airacId>{airac_id}</airacId>
    </airac>
  </queryCriteria>
</airspace:CompleteAIXMDatasetRequest>
""",
        "FlightListByAirspaceRequest": """
<fl:FlightListByAirspaceRequest
    xmlns:fl="eurocontrol/cfmu/b2b/FlightServices"
    xmlns:cm="eurocontrol/cfmu/b2b/CommonServices"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:fw="eurocontrol/cfmu/b2b/FlowServices"
    xmlns:as="eurocontrol/cfmu/b2b/AirspaceServices"
    xmlns:ns8="http://www.fixm.aero/base/4.0"
    xmlns:ns7="http://www.fixm.aero/flight/4.0"
    xmlns:ns13="http://www.fixm.aero/nm/1.0"
    xmlns:ns9="http://www.fixm.aero/base/4.1"
    xmlns:ns12="http://www.fixm.aero/eurextension/4.0"
    xmlns:ns11="http://www.w3.org/1999/xlink"
    xmlns:ns10="http://www.fixm.aero/flight/4.1"
    xmlns:ns14="http://www.fixm.aero/messaging/4.1">
    <sendTime>{send_time:%Y-%m-%d %H:%M:%S}</sendTime>
    <dataset>
        <type>OPERATIONAL</type>
    </dataset>
    <includeProposalFlights>false</includeProposalFlights>
    <includeForecastFlights>true</includeForecastFlights>
    <trafficType>DEMAND</trafficType>
    <trafficWindow>
        <wef>{start:%Y-%m-%d %H:%M}</wef>
        <unt>{stop:%Y-%m-%d %H:%M}</unt>
    </trafficWindow>
    {requestedFlightFields}
    <airspace>{airspace}</airspace>
</fl:FlightListByAirspaceRequest>
""",
        "FlightRetrievalRequest": """
<fl:FlightRetrievalRequest
    xmlns:fl="eurocontrol/cfmu/b2b/FlightServices"
    xmlns:cm="eurocontrol/cfmu/b2b/CommonServices"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:fw="eurocontrol/cfmu/b2b/FlowServices"
    xmlns:as="eurocontrol/cfmu/b2b/AirspaceServices"
    xmlns:ns8="http://www.fixm.aero/base/4.0"
    xmlns:ns7="http://www.fixm.aero/flight/4.0"
    xmlns:ns13="http://www.fixm.aero/nm/1.0"
    xmlns:ns9="http://www.fixm.aero/base/4.1"
    xmlns:ns12="http://www.fixm.aero/eurextension/4.0"
    xmlns:ns11="http://www.w3.org/1999/xlink"
    xmlns:ns10="http://www.fixm.aero/flight/4.1"
    xmlns:ns14="http://www.fixm.aero/messaging/4.1">
    <sendTime>{send_time:%Y-%m-%d %H:%M:%S}</sendTime>
    <dataset>
        <type>OPERATIONAL</type>
    </dataset>
    <includeProposalFlights>false</includeProposalFlights>
    <flightId>
          <id>{flight_id}</id>
          <!-- or keys -->
    </flightId>
    <requestedFlightDatasets>flightPlan</requestedFlightDatasets>
    <requestedDataFormat>NM_B2B</requestedDataFormat>
</fl:FlightRetrievalRequest>
""",
        "FlightPlanListRequest": """
<fl:FlightPlanListRequest
    xmlns:fl="eurocontrol/cfmu/b2b/FlightServices"
    xmlns:cm="eurocontrol/cfmu/b2b/CommonServices"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:fw="eurocontrol/cfmu/b2b/FlowServices"
    xmlns:as="eurocontrol/cfmu/b2b/AirspaceServices"
    xmlns:ns8="http://www.fixm.aero/base/4.0"
    xmlns:ns7="http://www.fixm.aero/flight/4.0"
    xmlns:ns13="http://www.fixm.aero/nm/1.0"
    xmlns:ns9="http://www.fixm.aero/base/4.1"
    xmlns:ns12="http://www.fixm.aero/eurextension/4.0"
    xmlns:ns11="http://www.w3.org/1999/xlink"
    xmlns:ns10="http://www.fixm.aero/flight/4.1"
    xmlns:ns14="http://www.fixm.aero/messaging/4.1">
    <sendTime>{send_time:%Y-%m-%d %H:%M:%S}</sendTime>
    <aircraftId>{aircraftId}</aircraftId>
    <aerodromeOfDeparture>{origin}</aerodromeOfDeparture>
    <nonICAOAerodromeOfDeparture>false</nonICAOAerodromeOfDeparture>
    <airFiled>false</airFiled>
    <aerodromeOfDestination>{destination}</aerodromeOfDestination>
    <nonICAOAerodromeOfDestination>false</nonICAOAerodromeOfDestination>
    <estimatedOffBlockTime>
        <wef>{start:%Y-%m-%d %H:%M}</wef>
        <unt>{stop:%Y-%m-%d %H:%M}</unt>
    </estimatedOffBlockTime>
    {requestedFlightFields}
</fl:FlightPlanListRequest>
""",
        "FlightListByAerodromeRequest": """
<fl:FlightListByAerodromeRequest
    xmlns:fl="eurocontrol/cfmu/b2b/FlightServices"
    xmlns:cm="eurocontrol/cfmu/b2b/CommonServices"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:fw="eurocontrol/cfmu/b2b/FlowServices"
    xmlns:as="eurocontrol/cfmu/b2b/AirspaceServices"
    xmlns:ns8="http://www.fixm.aero/base/4.0"
    xmlns:ns7="http://www.fixm.aero/flight/4.0"
    xmlns:ns13="http://www.fixm.aero/nm/1.0"
    xmlns:ns9="http://www.fixm.aero/base/4.1"
    xmlns:ns12="http://www.fixm.aero/eurextension/4.0"
    xmlns:ns11="http://www.w3.org/1999/xlink"
    xmlns:ns10="http://www.fixm.aero/flight/4.1"
    xmlns:ns14="http://www.fixm.aero/messaging/4.1">
    <sendTime>{send_time:%Y-%m-%d %H:%M:%S}</sendTime>
    <dataset>
        <type>OPERATIONAL</type>
    </dataset>
    <includeProposalFlights>false</includeProposalFlights>
    <includeForecastFlights>true</includeForecastFlights>
    <trafficType>DEMAND</trafficType>
    <trafficWindow>
        <wef>{start:%Y-%m-%d %H:%M}</wef>
        <unt>{stop:%Y-%m-%d %H:%M}</unt>
    </trafficWindow>
    {requestedFlightFields}
    <aerodrome>{aerodrome}</aerodrome>
    <aerodromeRole>{aerodromeRole}</aerodromeRole>
</fl:FlightListByAerodromeRequest>
""",
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
        res = self.post(data)
        assert res.reply is not None

        # There may be several dataset available.
        # For now, we keep the latest one
        latest = max(
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
