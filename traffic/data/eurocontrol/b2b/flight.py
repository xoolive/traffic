from __future__ import annotations

import re
import textwrap
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    NoReturn,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
from xml.dom import minidom
from xml.etree import ElementTree

import pandas as pd

from ....core import Flight, FlightPlan
from ....core.mixins import DataFrameMixin, _HBox
from ....core.time import timelike, to_datetime
from .reply import B2BReply
from .xml import REQUESTS

rename_cols = {
    "actualOffBlockTime": "AOBT",
    "actualTakeOffTime": "ATOT",
    "actualTimeOfArrival": "ATOA",
    "aerodromeOfDeparture": "origin",
    "aerodromeOfDestination": "destination",
    "aircraftAddress": "icao24",
    "aircraftId": "callsign",
    "aircraftType": "typecode",
    "calculatedOffBlockTime": "COBT",
    "calculatedTakeOffTime": "CTOT",
    "calculatedTimeOfArrival": "CTOA",
    "estimatedOffBlockTime": "EOBT",
    "estimatedTakeOffTime": "ETOT",
    "estimatedTimeOfArrival": "ETOA",
}

default_flight_fields: Set[str] = {
    "actualOffBlockTime",
    "actualTakeOffTime",
    "actualTimeOfArrival",
    "aircraftAddress",
    # "aircraftOperator",
    "aircraftType",
    # "alternateAerodromes",
    # "apiSubmissionRules",
    # "applicableScenarios",
    # "arrivalInformation",
    # "atfcmMeasureLocations",
    # "avoidedRegulations",
    "calculatedOffBlockTime",
    "calculatedTakeOffTime",
    "calculatedTimeOfArrival",
    # "caughtInHotspots",
    # "ccamsSSRCode",
    # "cdm",
    # "cdmEstimatedOffBlockTime",
    # "cfmuFlightType",
    # "compareWithOtherTrafficType",
    # "confirmedCTFM",
    # "ctfmAirspaceProfile",
    # "ctfmPointProfile",
    # "ctfmRequestedFlightLevels",
    # "ctfmRestrictionProfile",
    "ctotLimitReason",
    "ctotShiftAlreadyAppliedByTower",
    "currentDepartureTaxiTimeAndProcedure",
    # "defaultReroutingRequestedFlightLevel",
    # "defaultReroutingRequestedSpeed",
    "delay",
    "delayCharacteristics",
    "departureTolerance",
    "divertedAerodromeOfDestination",
    # "equipmentCapabilityAndStatus",
    # "estimatedElapsedTime",
    "estimatedTakeOffTime",
    "estimatedTimeOfArrival",
    "excludedRegulations",
    "exclusionFromRegulations",
    # "exemptedFromRegulations",
    # "famStatus",
    # "filedRegistrationMark",
    # "filingRule",
    # "flightCriticality",
    # "flightDataVersionNr",
    # "flightHistory",
    # "flightLevelAtReferenceLocationEntry",
    # "flightLevelAtReferenceLocationExit",
    # "flightState",
    # "ftfmAirspaceProfile",
    # "ftfmPointProfile",
    # "ftfmRequestedFlightLevels",
    # "ftfmRestrictionProfile",
    # "fuelConsumptionIndicator",
    # "hasBeenForced",
    # "hasOtherRegulations",
    # "highestModelAirspaceProfile",
    # "highestModelPointProfile",
    # "highestModelRestrictionProfile",
    # "hotspots",
    "icaoRoute",
    # "initialFPLMessageOriginator",
    # "isProposalFlight",
    # "lastATFMMessageOriginator",
    # "lastATFMMessageReceivedOrSent",
    # "lastATFMMessageType",
    # "lastFPLMessageOriginator",
    "lastKnownPosition",
    # "lateFiler",
    # "lateUpdater",
    # "mcdmInfo",
    # "minimumRequestedRVR",
    "mostPenalisingRegulation",
    "mostPenalisingRegulationCause",
    # "newRouteMinShiftDelayImprovement",
    "oceanicReroute",
    # "operatingAircraftOperator",
    # "profileValidity",
    # "proposalInformation",
    # "readyEstimatedOffBlockTime",
    "readyStatus",
    "regulationLocations",
    # "requestedFlightLevel",
    # "requestedInitialFlightLevel",
    # "requestedInitialSpeed",
    # "reroutable",
    # "reroutingIndicator",
    # "reroutingOpportunitiesExist",
    # "revisionTimes",
    # "routeChargeIndicator",
    "routeLength",
    # "rtfmAirspaceProfile",
    # "rtfmPointProfile",
    # "rtfmRequestedFlightLevels",
    # "rtfmRestrictionProfile",
    # "runwayVisualRange",
    # "slotIssued",
    # "slotSwapCounter",
    # "slotZone",
    # "suspensionInfo",
    # "suspensionStatus",
    # "targetTimeOverFix",
    "taxiTime",
    # "timeAtReferenceLocationEntry",
    # "timeAtReferenceLocationExit",
    # "trendAtReferenceLocationEntry",
    # "trendAtReferenceLocationExit",
    # "trendAtReferenceLocationMiddle",
    # "turnFlightForLocation",
    # "wakeTurbulenceCategory",
    # "worstLoadStateAtReferenceLocation",
    # "yoyoFlightForLocation",
}


class ParseFields:
    def __init__(self) -> None:
        self.route: Optional[str] = None

    def parse(self, elt: ElementTree.Element) -> Dict[str, Any]:
        if ("Time" in elt.tag or "time" in elt.tag) and elt.text is not None:
            return {elt.tag: pd.Timestamp(elt.text, tz="UTC")}
        if elt.text is not None:
            return {elt.tag: elt.text}
        method: Callable[..., Dict[str, Any]] = getattr(
            type(self), elt.tag, None
        )
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
    def from_file(cls, filename: str) -> "FlightInfo":
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
        """Returns the flight plan in ICAO format."""
        return FlightPlan(
            self.icaoRoute,
            self.aerodromeOfDeparture,
            self.aerodromeOfDestination,
        )

    @property
    def callsign(self) -> None | str:
        if hasattr(self, "aircraftId"):
            return self.aircraftId
        return None

    @property
    def icao24(self) -> None | str:
        if hasattr(self, "aircraftAddress"):
            return self.aircraftAddress.lower()
        return None

    def _repr_html_(self) -> str:
        from ....data import aircraft, airports

        aircraft_fmt = "<code>%icao24</code> · %flag %registration (%typecode)"
        title = f"<b>Flight {self.flight_id}</b> "
        title += "<ul>"
        if hasattr(self, "aircraftId"):
            title += f"<li><b>callsign:</b> {self.aircraftId}</li>"
        departure = airports[self.aerodromeOfDeparture]
        destination = airports[self.aerodromeOfDestination]
        title += f"<li><b>from:</b> {departure:%name (%icao/%iata)}</li>"
        title += f"<li><b>to:</b> {destination:%name (%icao/%iata)}</li>"
        if hasattr(self, "aircraftAddress"):
            ac = aircraft.get_unique(self.aircraftAddress.lower())
            title += "<li><b>aircraft:</b> {aircraft}</li>".format(
                aircraft=format(ac, aircraft_fmt)
            )

        cumul = list()
        cumul.append(
            pd.DataFrame.from_dict(
                [
                    dict(
                        (value, getattr(self, key, None))
                        for key, value in rename_cols.items()
                        if len(value) == 4
                    )
                ]
            ).T.rename(columns={0: self.flight_id})
        )

        no_wrap_div = '<div style="float: left; margin: 10px">{}</div>'
        fp = self.flight_plan
        fp_svg = fp._repr_svg_()
        return (
            title
            + "<br/><code>"
            + "<br/>".join(textwrap.wrap(re.sub(r"\s+", " ", fp.repr).strip()))
            + "</code><br/>"
            + (no_wrap_div.format(fp_svg) if fp_svg is not None else "")
            + _HBox(*cumul)._repr_html_()
        )

    def __getattr__(self, name: str) -> Union[str, pd.Timestamp]:
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

    def parsePlan(self, name: str) -> None | Flight:
        """
        If available, parse the FTFM (m1), RTFM (m2) or CTFM (m3) profiles.

        :param name: one of ftfmPointProfile, rtfmPointProfile, ctfmPointProfile

        """
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

    columns_options = dict(
        flightId=dict(style="blue bold"),
        callsign=dict(),
        icao24=dict(),
        typecode=dict(),
        origin=dict(),
        destination=dict(),
        EOBT=dict(),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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

    def __getitem__(self, item: str) -> None | FlightInfo:
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

        self.data = self.data.rename(columns=rename_cols).replace(
            "SLOT_TIME_NOT_LIMITED", ""
        )

        for feat in [
            "AOBT",
            "ATOA",
            "ATOT",
            "COBT",
            "CTOA",
            "CTOT",
            "EOBT",
            "ETOA",
            "ETOT",
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
    columns_options = dict(
        flightId=dict(style="blue bold"),
        callsign=dict(),
        origin=dict(),
        destination=dict(),
        EOBT=dict(),
        status=dict(),
    )

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

    def __getitem__(self, item: str) -> Optional[FlightInfo]:
        handle = next(
            (df for _, df in self.data.iterrows() if df.flightId == item), None
        )
        if handle is None:
            return None

        from ... import nm_b2b

        return nm_b2b.flight_get(
            EOBT=handle.EOBT,
            callsign=handle.callsign,
            origin=handle.origin,
            destination=handle.destination,
        )


class FlightManagement:
    def flight_search(
        self,
        start: None | timelike = None,
        stop: None | timelike = None,
        *args: Any,
        callsign: None | str = None,
        origin: None | str = None,
        destination: None | str = None,
    ) -> FlightPlanList:
        """Returns a **minimum set of information** about flights.

        :param start: (UTC), by default current time
        :param stop: (UTC), by default one hour later

        The method must take at least one of:

        :param callsign: (wildcard accepted)
        :param origin: flying from a given airport (ICAO 4 letter code).
        :param destination: flying to a given airport (ICAO 4 letter code).

        **Example usage:**

        .. jupyter-execute::

            # All KLM flights bound for Amsterdam Schiphol
            nm_b2b.flight_search(destination="EHAM", callsign="KLM*")

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
        self, EOBT: timelike, callsign: str, origin: str, destination: str
    ) -> FlightInfo:
        """Returns full information about a given flight.

        This method requires all parameters:

        :param EOBT: Estimated off-block time
        :param callsign: **NO** wildcard accepted
        :param origin: flying from a given airport (ICAO 4 letter code).
        :param destination: flying to a given airport (ICAO 4 letter code).

        This method is called when indexing the return value of a
        :meth:`~traffic.data.eurocontrol.b2b.NMB2B.flight_search`.

        """

        EOBT = to_datetime(EOBT)
        data = REQUESTS["FlightRetrievalRequest"].format(
            send_time=datetime.now(timezone.utc),
            callsign=callsign,
            origin=origin,
            destination=destination,
            eobt=EOBT,
        )
        rep = self.post(data)  # type: ignore
        return FlightInfo.fromET(rep.reply.find("data/flight"))

    def flight_list(
        self,
        start: None | timelike = None,
        stop: None | timelike = None,
        *args: Any,
        airspace: None | str = None,
        airport: None | str = None,
        origin: None | str = None,
        destination: None | str = None,
        regulation: None | str = None,
        fields: None | list[str] = None,
    ) -> None | FlightList:
        """Returns requested information about flights matching a criterion.

        :param start: (UTC), by default current time
        :param stop: (UTC), by default one hour later

        Exactly one of the following parameters must be passed:

        :param airspace: the name of an airspace aircraft fly through
        :param airport: flying from or to a given airport (ICAO 4 letter code).
        :param origin: flying from a given airport (ICAO 4 letter code).
        :param destination: flying to a given airport (ICAO 4 letter code).
        :param regulation: identifier of a regulation (see
            :meth:`~traffic.data.eurocontrol.b2b.NMB2B.regulation_list`)
        :param fields: additional fields to request. By default, a set of
            (arguably) relevant fields are requested.

        **Example usage:**

        .. jupyter-execute::

            # Get all flights scheduled out of Paris CDG
            nm_b2b.flight_list(origin="LFPG")

        **See also:**

            - :meth:`~traffic.data.eurocontrol.b2b.NMB2B.flight_get` returns
              full information about a given flight. It is also accessible by
              indexing a
              :class:`~traffic.data.eurocontrol.b2b.flight.FlightList` object
              with the ``flightId`` field.

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
