# fmt: off

import re
import textwrap
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any, Dict, List, NoReturn, Optional, Set, Type, TypeVar, Union
)
from xml.dom import minidom
from xml.etree import ElementTree

import pandas as pd

from ....core import Flight, FlightPlan
from ....core.mixins import DataFrameMixin, _HBox
from ....core.time import timelike, to_datetime
from .reply import B2BReply
from .xml import REQUESTS

# fmt: on

rename_cols = {
    "aircraftId": "callsign",
    "aircraftType": "typecode",
    "aerodromeOfDeparture": "origin",
    "aerodromeOfDestination": "destination",
    "estimatedOffBlockTime": "EOBT",
    "cdmEstimatedOffBlockTime": "cdmEOBT",
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
    "delay",
    "delayCharacteristics",
    "mostPenalisingRegulation",
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
                airway = airways[self.route]
                if airway is not None:
                    nx = navaids.extent(airway)
                    if nx is not None:
                        fix = nx[pointId.text]
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
    @classmethod
    def from_file(cls, filename: str):
        et = ElementTree.parse(filename)
        return cls.fromET(et.getroot())

    def to_xml(self, filename: Union[None, str, Path] = None) -> None:

        if isinstance(filename, str):
            filepath = Path(filename)

        if isinstance(filename, Path):
            filepath = filename

        if filename is None or filepath.is_dir():
            name = "{eobt:%Y-%m-%d}_{id_}_{callsign}_{from_}_{to}.xml"
        elif isinstance(filename, str):
            name = filename

        name = name.format(
            id_=self.flight_id,
            eobt=self.estimatedOffBlockTime,
            callsign=self.callsign,
            from_=self.aerodromeOfDeparture,
            to=self.aerodromeOfDestination,
        )

        if filepath.is_dir():
            filepath = filepath / name
        else:
            filepath = Path(name)

        ElementTree.ElementTree(self.reply).write(filepath)

    @property
    def flight_id(self) -> str:
        assert self.reply is not None
        elt = self.reply.find("flightId/id")
        assert elt is not None
        assert elt.text is not None
        return elt.text

    @property
    def flight_plan(self) -> FlightPlan:
        return FlightPlan(
            self.icaoRoute,
            self.aerodromeOfDeparture,
            self.aerodromeOfDestination,
        )

    @property
    def callsign(self) -> Optional[str]:
        if hasattr(self, "aircraftId"):
            return self.aircraftId
        return None

    @property
    def icao24(self) -> Optional[str]:
        if hasattr(self, "aircraftAddress"):
            return self.aircraftAddress.lower()
        return None

    def _repr_html_(self):
        from ....data import aircraft, airports

        title = f"<h4><b>Flight {self.flight_id}</b> "
        title += f"({self.aerodromeOfDeparture} → "
        title += f"{self.aerodromeOfDestination})</h4>"
        if hasattr(self, "aircraftId"):
            title += f"callsign: {self.aircraftId}<br/>"
        title += f" from {airports[self.aerodromeOfDeparture]}<br/>"
        title += f" to {airports[self.aerodromeOfDestination]}<br/><br/>"

        cumul = list()
        if hasattr(self, "aircraftAddress"):
            cumul.append(aircraft[self.aircraftAddress.lower()].T)

        cumul.append(
            pd.DataFrame.from_dict(
                [
                    {
                        "EOBT": self.estimatedOffBlockTime
                        if hasattr(self, "estimatedOffBlockTime")
                        else None,
                        "ETOT": self.estimatedTakeOffTime
                        if hasattr(self, "estimatedTakeOffTime")
                        else None,
                        "ATOT": self.actualTakeOffTime
                        if hasattr(self, "actualTakeOffTime")
                        else None,
                        "ETOA": self.estimatedTimeOfArrival
                        if hasattr(self, "estimatedTimeOfArrival")
                        else None,
                        "ATOA": self.actualTimeOfArrival
                        if hasattr(self, "actualTimeOfArrival")
                        else None,
                    }
                ]
            ).T.rename(columns={0: self.flight_id})
        )

        no_wrap_div = '<div style="float: left; margin: 10px">{}</div>'
        fp = self.flight_plan
        return (
            title
            + "<br/><code>"
            + "<br/>".join(textwrap.wrap(re.sub(r"\s+", " ", fp.repr).strip()))
            + "</code><br/>"
            + no_wrap_div.format(fp._repr_svg_())
            + _HBox(*cumul)._repr_html_()
        )

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
    def fromB2BReply(
        cls: Type[FlightListTypeVar], r: B2BReply
    ) -> FlightListTypeVar:
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

        self.format_data()

    def format_data(self) -> None:

        if "nonICAOAerodromeOfDeparture" in self.data.columns:
            self.data = self.data.drop(
                columns=[
                    "nonICAOAerodromeOfDeparture",
                    "nonICAOAerodromeOfDestination",
                    "airFiled",
                ]
            )

        self.data = self.data.rename(columns=rename_cols)

        for feat in [
            "EOBT",
            "ETOT",
            "CTOT",
            "ATOT",
            "ETOA",
            "CTOA",
            "ATOA",
            "cdmEOBT",
        ]:
            if feat in self.data.columns:
                self.data = self.data.assign(
                    **{
                        feat: self.data[feat].apply(
                            lambda x: pd.Timestamp(x, tz="utc")
                        )
                    }
                )

        for feat in ["currentlyUsedTaxiTime", "taxiTime", "delay"]:
            if feat in self.data.columns:
                self.data = self.data.assign(
                    **{
                        feat: self.data[feat].apply(
                            lambda x: pd.Timedelta(
                                f"{x[:2]} hours {x[2:4]} minutes "
                                + f"{x[4:6]} seconds"
                                if feat == "currentlyUsedTaxiTime"
                                else ""
                            )
                            if x == x
                            else pd.Timedelta("0")
                        )
                    }
                )

        if "icao24" in self.data.columns:
            self.data = self.data.assign(icao24=self.data.icao24.str.lower())

        if "EOBT" in self.data.columns:
            self.data = self.data.sort_values("EOBT")


class FlightPlanList(FlightList):
    def build_df(self) -> None:
        assert self.reply is not None

        self.data = pd.DataFrame.from_records(
            [
                {
                    **{
                        "flightId": elt.find("id/id").text  # type: ignore
                        if elt.find("id/id") is not None
                        else None
                    },
                    **{
                        p.tag: p.text
                        for p in elt.find("id/keys")  # type: ignore
                    },
                    **{
                        p.tag: p.text
                        for p in elt
                        if p.tag != "flightId" and p.text is not None
                    },
                }
                for elt in self.reply.findall("summaries/lastValidFlightPlan")
            ]
        )

        self.format_data()

    def __getitem__(self, item) -> Optional[FlightInfo]:
        handle = next(
            (df for _, df in self.data.iterrows() if df.flightId == item), None
        )
        if handle is None:
            return None

        from ... import nm_b2b

        return nm_b2b.flight_get(
            eobt=handle.EOBT,
            callsign=handle.callsign,
            origin=handle.origin,
            destination=handle.destination,
        )


class FlightManagement:
    def flight_search(
        self,
        start: Optional[timelike] = None,
        stop: Optional[timelike] = None,
        *args,
        callsign: Optional[str] = None,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
    ) -> FlightPlanList:
        """Returns a **minimum set of information** about flights.

        By default:

        - the start parameter takes the current time.
        - the stop parameter is one hour after the start parameter.

        The method must take at least one of:

        - callsign (wildcard accepted)
        - origin airport (ICAO 4 letter code)
        - destination airport (ICAO 4 letter code)

        The return type has a representation similar to a pandas
        DataFrame and may be indexed by a flight_id in order to
        request full information about that flight.

        **Example usage:**

        .. code:: python

            # All KLM flights out of Amsterdam Schiphol
            res = nm_b2b.flight_search(origin="EHAM", callsign="KLM*")

            # All flights in a given day going to Kansai International Airport
            res = nm_b2b.flight_search(
                start="2019-12-22 00:00",
                stop="2019-12-23 00:00",
                destination="RJBB"
            )

            # Get full information about one particular flight
            flight_info = res["AT02478340"]

        **See also:**

            - `flight_list
              <#traffic.data.eurocontrol.b2b.NMB2B.flight_list>`_ which
              returns more comprehensive information about flights.

        """

        if start is not None:
            start = to_datetime(start)
        else:
            start = datetime.now(timezone.utc)

        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(hours=1)

        data = REQUESTS["FlightPlanListRequest"].format(
            send_time=datetime.now(timezone.utc),
            aircraftId=callsign if callsign is not None else "*",
            origin=origin if origin is not None else "*",
            destination=destination if destination is not None else "*",
            start=start,
            stop=stop,
        )

        rep = self.post(data)  # type: ignore
        return FlightPlanList.fromET(rep.reply.find("data"))

    def flight_get(
        self, eobt: timelike, callsign: str, origin: str, destination: str
    ) -> FlightInfo:
        """Returns full information about a given flight.

        This method requires all parameters:

        - eobt: Estimated off-block time (as string, numeral or timestamp)
        - callsign (**NO** wildcard accepted)
        - origin airport (ICAO 4 letter code)
        - destination airport (ICAO 4 letter code)

        It is recommended to use this method through an indexation
        on the return of a `flight_search
        <#traffic.data.eurocontrol.b2b.NMB2B.flight_search>`_.

        """

        eobt = to_datetime(eobt)
        data = REQUESTS["FlightRetrievalRequest"].format(
            send_time=datetime.now(timezone.utc),
            callsign=callsign,
            origin=origin,
            destination=destination,
            eobt=eobt,
        )
        rep = self.post(data)  # type: ignore
        return FlightInfo.fromET(rep.reply.find("data/flight"))

    def flight_list(
        self,
        start: Optional[timelike] = None,
        stop: Optional[timelike] = None,
        *args,
        airspace: Optional[str] = None,
        airport: Optional[str] = None,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        regulation: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Optional[FlightList]:
        """Returns requested information about flights matching a criterion.

        By default:

        - the start parameter takes the current time.
        - the stop parameter is one hour after the start parameter.

        Exactly one of the following parameters must be passed:

        - airspace: returns all flights going through a given airspace.
        - airport: returns all flights flying from or to a given airport
          (ICAO 4 letter code).
        - origin: returns all flights flying from a given airport
          (ICAO 4 letter code).
        - destination: returns all flights flying to a given airport
          (ICAO 4 letter code).
        - regulation: returns all flights impacted by a given regulation.

        By default, a set of (arguably) relevant fields are requested. More
        fields can be requested when passed to the field parameter.

        **Example usage:**

        .. code:: python

            # Get all flights scheduled out of Paris CDG in a time frame.
            res = nm_b2b.flight_list(
                "2019-12-22 10:00",
                "2019-12-22 10:30",
                origin="LFPG",
                fields=["aircraftOperator", "ctotLimitReason"],
            )

            # Get **requested** information about one particular flight
            flightinfo = res["AT02474519"]

        **See also:**

            - `flight_get
              <#traffic.data.eurocontrol.b2b.NMB2B.flight_get>`_ which
              returns full information about a given flight.

        """

        if start is not None:
            start = to_datetime(start)
        else:
            start = datetime.now(timezone.utc)

        if stop is not None:
            stop = to_datetime(stop)
        else:
            stop = start + timedelta(hours=1)

        msg = """At most one parameter must be set among:
- airspace
- airport (or origin, or destination)
        """
        query = [airspace, airport, origin, destination, regulation]
        if sum(x is not None for x in query) > 1:
            raise RuntimeError(msg)

        _fields = fields if fields is not None else []
        if airspace is not None:
            data = REQUESTS["FlightListByAirspaceRequest"].format(
                send_time=datetime.now(timezone.utc),
                start=start,
                stop=stop,
                requestedFlightFields="\n".join(
                    f"<requestedFlightFields>{field}</requestedFlightFields>"
                    for field in default_flight_fields.union(_fields)
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
                    for field in default_flight_fields.union(_fields)
                ),
                aerodrome=airport,
                aerodromeRole=role,
            )
            rep = self.post(data)  # type: ignore
            return FlightList.fromB2BReply(rep)

        if regulation is not None:
            data = REQUESTS["FlightListByMeasureRequest"].format(
                send_time=datetime.now(timezone.utc),
                start=start,
                stop=stop,
                requestedFlightFields="\n".join(
                    f"<requestedFlightFields>{field}</requestedFlightFields>"
                    for field in default_flight_fields.union(_fields)
                ),
                regulation=regulation,
            )
            rep = self.post(data)  # type: ignore
            return FlightList.fromB2BReply(rep)

        return None
