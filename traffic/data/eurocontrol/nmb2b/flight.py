from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set, Type, TypeVar
from xml.etree import ElementTree

import pandas as pd

from ....core.mixins import DataFrameMixin
from ....core.time import timelike, to_datetime
from .reply import B2BReply
from .xml import REQUESTS

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


def requestedFlightFields(fields: List[str]) -> str:
    return "\n".join(
        "<requestedFlightFields>" + field + "</requestedFlightFields>"
        for field in fields
    )


class Flight(B2BReply):
    def __getattr__(self, name) -> str:
        cls = type(self)
        assert self.reply is not None
        elt = self.reply.find(name)
        if elt is not None and elt.text is not None:
            return elt.text
        msg = "{.__name__!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls, name))


# https://github.com/python/mypy/issues/2511
FlightListTypeVar = TypeVar("FlightListTypeVar", bound="FlightList")


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
    def fromET(
        cls: Type[FlightListTypeVar], tree: ElementTree.Element
    ) -> FlightListTypeVar:
        instance = cls()
        instance.reply = tree
        instance.build_df()
        return instance

    def __getitem__(self, item) -> Optional[Flight]:
        assert self.reply is not None
        for elt in self.reply.findall("data/flights/flight"):
            key = elt.find("flightId/id")
            assert key is not None
            if key.text == item:
                return Flight.fromET(elt)

        return None

    def _ipython_key_completions_(self) -> Set[str]:
        return set(self.data.flightId.unique())

    def build_df(self) -> None:
        assert self.reply is not None

        self.data = pd.DataFrame.from_records(
            [
                {
                    **{
                        "flightId": elt.find("flightId/id").text  # type: ignore
                    },
                    **{
                        p.tag: p.text
                        for p in elt.find("flightId/keys")  # type: ignore
                    },
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


class FlightManagement:

    FM_REQUESTS = {
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
    </flightId>
    {requestedFlightFields}
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
    }

    def list_flight(
        self,
        start: timelike,
        stop: Optional[timelike] = None,
        airspace: Optional[str] = None,
        airport: Optional[str] = None,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        fields: List[str] = [],
    ) -> Optional[FlightList]:

        start = to_datetime(start)
        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(hours=2)

        if airspace is not None:
            data = REQUESTS["FlightListByAirspaceRequest"].format(
                send_time=datetime.now(timezone.utc),
                start=start,
                stop=stop,
                requestedFlightFields=requestedFlightFields(fields),
                airspace=airspace,
            )
            rep = self.post(data)  # type: ignore
            return FlightList.fromB2BReply(rep)

        if airport is not None or origin is not None or destination is not None:
            role = "BOTH"
            if origin is not None:
                airport = origin
                role = "DEPARTURE"
            if destination is not None:
                airport = destination
                role = "ARRIVAL"

            data = REQUESTS["FlightListByAerodromeRequest"].format(
                send_time=datetime.now(timezone.utc),
                start=start,
                stop=stop,
                requestedFlightFields=requestedFlightFields(fields),
                aerodrome=airport,
                aerodromeRole=role,
            )
            rep = self.post(data)  # type: ignore
            return FlightList.fromB2BReply(rep)

        return None
