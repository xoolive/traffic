import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NoReturn, Optional, Set, Type, TypeVar
from xml.dom import minidom
from xml.etree import ElementTree

import pandas as pd

from ....core import Flight
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


default_flight_fields: Set[str] = {
    "divertedAerodromeOfDestination",
    "readyEstimatedOffBlockTime",
    "cdmEstimatedOffBlockTime",
    "aircraftType",
    "estimatedTakeOffTime",
    "calculatedTakeOffTime",
    "actualTakeOffTime",
    "ctotShiftAlreadyAppliedByTower",
    "taxiTime",
    "currentlyUsedTaxiTime",
    "revisionTimes",
    "estimatedTimeOfArrival",
    "calculatedTimeOfArrival",
    "actualTimeOfArrival",
    # "requestedFlightLevel",
    # "timeAtReferenceLocationEntry",
    # "timeAtReferenceLocationExit",
    # "flightLevelAtReferenceLocationEntry",
    # "flightLevelAtReferenceLocationExit",
    # "trendAtReferenceLocationEntry",
    # "trendAtReferenceLocationExit",
    # "trendAtReferenceLocationMiddle",
    # "lateFiler",
    # "lateUpdater",
    # "suspensionStatus",
    # "suspensionResponseBy",
    # "exclusionFromRegulations",
    # "famStatus",
    # "readyStatus",
    # "aircraftOperator",
    # "operatingAircraftOperator",
    # "reroutingIndicator",
    # "newRouteMinShiftDelayImprovement",
    # "reroutable",
    # "reroutingOpportunitiesExist",
    # "cdm",
    # "slotIssued",
    # "slotImprovementProposal",
    # "exemptedFromRegulations",
    # "delay",
    # "delayCharacteristics",
    # "mostPenalisingRegulation",
    # "hasOtherRegulations",
    # "regulationLocations",
    # "atfcmMeasureLocations",
    # "lastATFMMessageType",
    # "lastATFMMessageReceivedOrSent",
    # "runwayVisualRange",
    # "confirmedCTFM",
    # "requestedInitialFlightLevel",
    # "requestedInitialSpeed",
    # "estimatedElapsedTime",
    # "filingRule",
    # "initialFPLMessageOriginator",
    # "lastFPLMessageOriginator",
    "icaoRoute",
    "routeLength",
    # "reroutingReference",
    # "defaultReroutingRequestedFlightLevel",
    # "defaultReroutingRequestedSpeed",
    # "departureTolerance",
    # "mostPenalisingRegulationCause",
    # "lastATFMMessageOriginator",
    # "ftfmPointProfile",
    # "rtfmPointProfile",
    # "ctfmPointProfile",
    # "ftfmAirspaceProfile",
    # "rtfmAirspaceProfile",
    # "ctfmAirspaceProfile",
    # "ftfmRequestedFlightLevels",
    # "rtfmRequestedFlightLevels",
    # "ctfmRequestedFlightLevels",
    # "flightHistory",
    # "equipmentCapabilityAndStatus",
    # "ftfmRestrictionProfile",
    # "rtfmRestrictionProfile",
    # "ctfmRestrictionProfile",
    # "cfmuFlightType",
    # "ccamsSSRCode",
    # "filedRegistrationMark",
    # "isProposalFlight",
    # "proposalExists",
    # "hasBeenForced",
    # "caughtInHotspots",
    # "hotspots",
    # "mcdmInfo",
    # "worstLoadStateAtReferenceLocation",
    # "compareWithOtherTrafficType",
    # "ctotLimitReason",
    # "profileValidity",
    # "targetTimeOverFix",
    # "flightState",
    "lastKnownPosition",
    # "highestModelPointProfile",
    # "highestModelAirspaceProfile",
    # "highestModelRestrictionProfile",
    # "slotSwapCounter",
    "aircraftAddress",
    # "arrivalInformation",
    # "slotZone",
    # "flightDataVersionNr",
    # "applicableScenarios",
    # "apiSubmissionRules",
    # "avoidedRegulations",
    # "routeChargeIndicator",
    # "fuelConsumptionIndicator",
    # "excludedRegulations",
}


class ParseFields:
    def __init__(self):
        self.route: Optional[str] = None

    def parse(self, elt: ElementTree.Element) -> Dict[str, Any]:
        if ("Time" in elt.tag or "time" in elt.tag) and elt.text is not None:
            return {elt.tag: pd.Timestamp(elt.text, tz="UTC")}
        if elt.text is not None:
            return {elt.tag: elt.text}
        method = getattr(type(self), elt.tag, None)
        if method is None:
            self.unknown(elt)
        return method(self, elt)

    def unknown(self, elt: ElementTree.Element) -> NoReturn:
        s = ElementTree.tostring(elt)
        raise Exception(minidom.parseString(s).toprettyxml(indent="  "))

    def flightLevel(self, point: ElementTree.Element) -> Dict[str, Any]:
        level = point.find("level")
        unit = point.find("unit")
        if level is not None and unit is not None:
            if unit.text == "F" and level.text is not None:
                return {"altitude": 100 * int(level.text)}

        self.unknown(point)

    def associatedRouteOrTerminalProcedure(
        self, point: ElementTree.Element
    ) -> Dict[str, Any]:
        sid = point.find("SID")
        star = point.find("STAR")
        route = point.find("route")
        if sid is not None:
            self.route = None
            id_ = sid.find("id")
            aerodrome = sid.find("aerodromeId")
            return {
                "route": id_.text if id_ is not None else None,
                "aerodrome": aerodrome.text if aerodrome is not None else None,
            }
        elif star is not None:
            self.route = None
            id_ = star.find("id")
            aerodrome = star.find("aerodromeId")
            return {
                "route": id_.text if id_ is not None else None,
                "aerodrome": aerodrome.text if aerodrome is not None else None,
            }
        elif route is not None:
            self.route = route.text
            return {"route": route.text}
        elif point.find("DCT") is not None:
            return {"route": "DCT"}

        self.unknown(point)

    def point(self, point: ElementTree.Element) -> Dict[str, Any]:
        pointId = point.find("pointId")
        if pointId is not None:
            from ....data import airways, navaids

            rep: Dict[str, Any] = {"FIX": pointId.text}
            if self.route is not None:
                fix = navaids.extent(airways[self.route])[pointId.text]
                if fix is not None:
                    rep["latitude"] = fix.latitude
                    rep["longitude"] = fix.longitude
            return rep
        dbePoint = point.find("nonPublishedPoint-DBEPoint")
        if dbePoint is not None:
            return {"FIX": dbePoint.text}
        geopoint = point.find("nonPublishedPoint-GeoPoint")
        if geopoint is not None:
            angle = geopoint.find("position/latitude/angle")
            side = geopoint.find("position/latitude/side")
            assert angle is not None and side is not None
            lat = int(angle.text) / 10000  # type: ignore
            if side.text == "SOUTH":
                lat *= -1

            angle = geopoint.find("position/longitude/angle")
            side = geopoint.find("position/longitude/side")
            assert angle is not None and side is not None
            lon = int(angle.text) / 10000  # type: ignore
            if side.text == "WEST":
                lat *= -1

            return {"latitude": lat, "longitude": lon}

        return self.unknown(point)


class FlightInfo(B2BReply):
    @property
    def flight_id(self) -> str:
        assert self.reply is not None
        elt = self.reply.find("flightId/id")
        assert elt is not None
        assert elt.text is not None
        return elt.text

    def __getattr__(self, name) -> str:
        cls = type(self)
        assert self.reply is not None
        elt = self.reply.find(name)
        if elt is None:
            elt = self.reply.find("flightId/keys/" + name)
        if elt is not None and elt.text is not None:
            if "Time" in name or "time" in name:
                return pd.Timestamp(elt.text, tz="UTC")
            return elt.text
        msg = "{.__name__!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls, name))

    def parsePlan(self, name) -> Optional[Flight]:
        assert self.reply is not None
        msg = "No {} found in requested fields"
        if self.reply.find(name) is None:
            warnings.warn(msg.format(name))
            return None
        parser = ParseFields()
        return Flight(
            pd.DataFrame.from_records(
                [
                    dict(elt for p in point for elt in parser.parse(p).items())
                    for point in self.reply.findall(name)
                ]
            )
            .rename(columns={"timeOver": "timestamp"})
            .assign(
                flightPlanPoint=lambda x: x.flightPlanPoint == "true",
                icao24=self.aircraftAddress.lower()
                if hasattr(self, "aircraftAddress")
                else None,
                callsign=self.aircraftId,
                origin=self.aerodromeOfDeparture,
                destination=self.aerodromeOfDestination,
                flight_id=self.flight_id,
                EOBT=self.estimatedOffBlockTime,
                typecode=self.aircraftType,
            )
        )


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

    def __getitem__(self, item) -> Optional[FlightInfo]:
        assert self.reply is not None
        for elt in self.reply.findall("data/flights/flight"):
            key = elt.find("flightId/id")
            assert key is not None
            if key.text == item:
                return FlightInfo.fromET(elt)

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
                        if elt.find("flightId/id") is not None
                        else None
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
            stop = start + timedelta(hours=1)

        msg = """At most one parameter must be set among:
- airspace
- airport (or origin, or destination)
        """
        query = [airspace, airport, origin, destination]
        if sum(x is not None for x in query) > 1:
            raise RuntimeError(msg)

        if airspace is not None:
            data = REQUESTS["FlightListByAirspaceRequest"].format(
                send_time=datetime.now(timezone.utc),
                start=start,
                stop=stop,
                requestedFlightFields="\n".join(
                    f"<requestedFlightFields>{field}</requestedFlightFields>"
                    for field in default_flight_fields.union(fields)
                ),
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
                requestedFlightFields="\n".join(
                    f"<requestedFlightFields>{field}</requestedFlightFields>"
                    for field in default_flight_fields.union(fields)
                ),
                aerodrome=airport,
                aerodromeRole=role,
            )
            rep = self.post(data)  # type: ignore
            return FlightList.fromB2BReply(rep)

        return None
