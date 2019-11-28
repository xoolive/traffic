import re
import textwrap
import warnings
from typing import Any, List, Optional, Union, cast

from shapely.geometry import LineString
from shapely.ops import linemerge

from ..data.basic.airports import Airport
from ..data.basic.airways import Route
from ..data.basic.navaid import Navaid
from .mixins import ShapelyMixin


class _ElementaryBlock:

    pattern: str

    def __init__(self, *args) -> None:
        self.elt = args

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.elt}"

    def get(self):
        return self

    @property
    def name(self) -> str:
        return (
            self.elt[0] if isinstance(self.elt[0], str) else repr(self.elt[0])
        )

    @classmethod
    def valid(cls, elt: str) -> bool:
        return bool(re.match(cls.pattern, elt))

    @classmethod
    def parse(
        cls,
        elt: str,
        previous_elt: Optional["_ElementaryBlock"] = None,
        is_last_elt: bool = False,
    ) -> Optional["_ElementaryBlock"]:

        if Direct.valid(elt):
            return Direct(elt)

        if previous_elt is None:
            if SpeedLevel.valid(elt):
                return SpeedLevel(elt)
            else:
                return Airway(elt)

        if is_last_elt:
            if Point.valid(elt):
                return Point(elt)
            elif STAR.valid(elt):
                return STAR(elt)

        elif ChangeOfFlightRule.valid(elt):
            return ChangeOfFlightRule(elt)

        elif SpeedLevelChangePoint.valid(elt):
            p, sl, *_ = elt.split("/")
            if Point.valid(p):
                return SpeedLevelChangePoint(Point(p), SpeedLevel(sl))
            elif CoordinatePoint.valid(p):
                return SpeedLevelChangePoint(CoordinatePoint(p), SpeedLevel(sl))

        elif CoordinatePoint.valid(elt):
            return CoordinatePoint(elt)

        elif type(previous_elt) in (Direct, Airway, SpeedLevel, SID):
            if Point.valid(elt):
                return Point(elt)
            elif SID.valid(elt):
                return SID(elt)

        elif Airway.valid(elt):
            return Airway(elt)

        return None


class Direct(_ElementaryBlock):
    pattern = "DCT$"


class SpeedLevel(_ElementaryBlock):
    pattern = r"([K,N]\d{4}|M\d{3})(([A,F]\d{3})|[S,M]\d{4})$"

    # -- Speed units --
    # K (Kilometers)
    # N (Knots)
    # M (Mach)
    # -- Altitude units --
    # F (Flight Level)
    # S (Standard Metric)
    # A (Altitude in feet)
    # M (Altitude in meter)

    def __init__(self, elt):
        x = re.match(self.pattern, elt)
        assert x is not None

        speed, alt = x.group(1), x.group(2)
        self.speed_unit = speed[0]
        self.speed = int(speed[1:])
        self.altitude_unit = alt[0]
        self.altitude = int(alt[1:])

        self.elt = speed, alt


class Airway(_ElementaryBlock):
    pattern = r"\w{2,7}$"

    def get(self) -> Optional[Route]:
        from traffic.data import nm_airways, airways

        res = nm_airways[self.elt[0]]
        if res is not None:
            return res
        return airways[self.elt[0]]


class Point(_ElementaryBlock):
    pattern = r"\D{2,5}$"

    def get(self) -> Optional[Navaid]:
        from traffic.data import nm_navaids, navaids

        if nm_navaids is not None:
            res = nm_navaids[self.elt[0]]
            if res is not None:
                return res
        return navaids[self.elt[0]]


class SID(Airway):
    def get(self, airport: Optional[str] = None) -> Optional[Route]:
        from traffic.data import nm_airways

        if airport is not None:
            return nm_airways[self.elt[0] + airport]

        possible = set(
            nm_airways.query(
                f'route.str.startswith("{self.elt[0]}")'
            ).data.route
        )
        if len(possible) == 1:
            return nm_airways[possible.pop()]

        warnings.warn(f"Could not find any corresponding SID for {self.elt[0]}")
        return None


class STAR(Airway):
    def get(self, airport: Optional[str] = None) -> Optional[Route]:
        from traffic.data import nm_airways

        if airport is not None:
            return nm_airways[self.elt[0] + airport]

        possible = set(
            nm_airways.query(
                f'route.str.startswith("{self.elt[0]}")'
            ).data.route
        )
        if len(possible) == 1:
            return nm_airways[possible.pop()]

        warnings.warn(
            f"Could not find any corresponding STAR for {self.elt[0]}"
        )
        return None


class ChangeOfFlightRule(_ElementaryBlock):
    pattern = r"VFR$|IFR$"


class CoordinatePoint(_ElementaryBlock):
    pattern = r"(\d{2}|\d{4})([N,S])(\d{3}|\d{5})([E,W])$"
    lon: float
    lat: float

    def get(self) -> Navaid:
        return Navaid(
            cast(str, self.elt),
            "NDB",
            self.lat,
            self.lon,
            None,
            None,
            None,
            None,
        )

    def __init__(self, elt):
        x = re.match(self.pattern, elt)
        assert x is not None
        lat, lat_sign = x.group(1), 1 if x.group(2) == "N" else -1
        lon, lon_sign = x.group(3), 1 if x.group(4) == "E" else -1

        if len(lat) == 2:
            self.lat = lat_sign * int(lat)
        else:
            self.lat = lat_sign * int(lat) / 100

        if len(lon) == 3:
            self.lon = lon_sign * int(lon)
        else:
            self.lon = lon_sign * int(lon) / 100

        self.elt = elt


class SpeedLevelChangePoint(_ElementaryBlock):

    pattern = "(.*)/(.*)"

    def get(self) -> Optional[Navaid]:
        return self.elt[0].get()

    @property
    def name(self) -> str:
        return self.elt[0].name

    @property
    def text(self) -> str:
        return f"{self.elt[0].name} ({self.elt[1].name})"


class FlightPlan(ShapelyMixin):

    origin: Optional[str]
    destination: Optional[str]

    def __init__(
        self,
        fp: str,
        origin: Union[None, str, Airport] = None,
        destination: Union[None, str, Airport] = None,
    ):
        self.repr = fp

        if isinstance(origin, Airport):
            self.origin = origin.icao
        else:
            self.origin = origin

        if isinstance(destination, Airport):
            self.destination = destination.icao
        else:
            self.destination = destination

    def __repr__(self) -> str:
        return "\n".join(textwrap.wrap(re.sub(r"\s+", " ", self.repr).strip()))

    def _info_html(self) -> str:
        from traffic.data import airports

        title = (
            "<h4><b>FlightPlan{from_}{to_}</b></h4>"
            "<div style='max-width: 600px'><code>"
        ).format(
            from_=f" from {airports[self.origin]}"
            if self.origin is not None
            else "",
            to_=f" to {airports[self.destination]}"
            if self.destination is not None
            else "",
        )
        title += " ".join(re.sub(r"\s+", " ", self.repr).strip().split())
        title += "</code></div>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def _repr_svg_(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super()._repr_svg_()

    def decompose(self) -> List[Optional[_ElementaryBlock]]:
        parsed: List[Optional[_ElementaryBlock]] = []
        blocks = repr(self).strip().split()

        for i, elt in enumerate(blocks):
            parsed.append(
                _ElementaryBlock.parse(
                    elt, parsed[-1] if i > 0 else None, len(blocks) - i == 1
                )
            )

        return parsed

    @property
    def shape(self) -> LineString:
        cumul: List[Any] = list()  # List[ShapelyMixin] ?
        elts = self.decompose()

        for i, e in enumerate(elts):
            if isinstance(e, Airway):
                if isinstance(e, SID):
                    cumul.append(e.get(self.origin))
                elif isinstance(e, STAR):
                    cumul.append(e.get(self.destination))
                else:
                    handle = e.get()
                    if handle is None:
                        warnings.warn(f"Missing information about {elts[i]}")
                        continue

                    previous, next_ = elts[i - 1], elts[i + 1]
                    if previous is None or next_ is None:
                        warnings.warn(f"Missing information around {elts[i]}")
                        continue

                    try:
                        cumul.append(handle[previous.name, next_.name])
                    except Exception as ex:
                        warnings.warn(
                            f"Missing information around {elts[i]}: {ex}"
                        )
                        continue

            if isinstance(e, Direct):
                from traffic.data import navaids

                previous, next_ = elts[i - 1], elts[i + 1]
                if previous is None or next_ is None:
                    warnings.warn(f"Missing information around {elts[i]}")
                    continue

                p1, p2 = navaids[previous.name], navaids[next_.name]
                if p1 is None or p2 is None:
                    warnings.warn(
                        f"Could not find {previous.name} or {next_.name}"
                    )
                    continue
                coords = [(p1.lon, p1.lat), (p2.lon, p2.lat)]
                cumul.append(
                    Route(
                        LineString(coordinates=coords),
                        "DCT",
                        [previous.name, next_.name],
                    )
                )

        return linemerge([x.shape for x in cumul if x is not None])
