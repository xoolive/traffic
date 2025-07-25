import re
import textwrap
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

from shapely.geometry import LineString
from shapely.ops import linemerge

from .mixins import PointBase, ShapelyMixin
from .structure import Airport, Navaid, Route

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from ipyleaflet import Polyline as LeafletPolyline
    from matplotlib.artist import Artist


class _ElementaryBlock:
    pattern: str

    def __init__(self, *args: Union[None, str, "_ElementaryBlock"]) -> None:
        self.elt = args

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.elt}"

    def get(self) -> "_ElementaryBlock":
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

        elif isinstance(previous_elt, Point) and DirectPoint.valid(elt):
            # EXPERIMENTAL
            return DirectPoint(elt)

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

    def __init__(self, elt: str) -> None:
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

    def get(self) -> Optional[Route]:  # type: ignore
        from traffic.data import airways

        if not isinstance(self.elt[0], str):
            return None

        return airways.get(self.elt[0])

    @classmethod
    def valid(cls, elt: str) -> bool:
        return bool(re.match(cls.pattern, elt)) and (
            any(i.isdigit() for i in elt)
            # North Atlantic tracks
            or elt.startswith("NAT")
        )


class Point(_ElementaryBlock):
    pattern = r"\D{2,5}$"

    def get(self) -> Optional[Navaid]:  # type: ignore
        from traffic.data import navaids

        if not isinstance(self.elt[0], str):
            return None

        return navaids.get(self.elt[0])


# EXPERIMENTAL
class DirectPoint(Point):
    pass


class SID(Airway):
    @classmethod
    def valid(cls, elt: str) -> bool:
        return bool(re.match(cls.pattern, elt))

    def get(  # type: ignore
        self, airport: Optional[str] = None
    ) -> Optional[Route]:
        from traffic.data import airways, nm_airways

        if not isinstance(self.elt[0], str):
            return None

        if airport is not None:
            return airways.get(self.elt[0] + airport)

        if nm_airways.available:
            possible = set(
                nm_airways.data.query(
                    f'route.str.startswith("{self.elt[0]}")'
                ).route
            )
            if len(possible) == 1:
                return nm_airways[possible.pop()]

        warnings.warn(f"Could not find any corresponding SID for {self.elt[0]}")
        return None


class STAR(Airway):
    @classmethod
    def valid(cls, elt: str) -> bool:
        return bool(re.match(cls.pattern, elt))

    def get(  # type: ignore
        self, airport: Optional[str] = None
    ) -> Optional[Route]:
        from traffic.data import airways, nm_airways

        if not isinstance(self.elt[0], str):
            return None

        if airport is not None:
            return airways.get(self.elt[0] + airport)

        if nm_airways.available:
            possible = set(
                nm_airways.data.query(
                    f'route.str.startswith("{self.elt[0]}")'
                ).route
            )
            if len(possible) == 1:
                return nm_airways[possible.pop()]

        warnings.warn(f"Could not find any corresponding SID for {self.elt[0]}")
        return None


class ChangeOfFlightRule(_ElementaryBlock):
    pattern = r"VFR$|IFR$"


class CoordinatePoint(_ElementaryBlock):
    pattern = r"(\d{2}|\d{4})([N,S])(\d{3}|\d{5})([E,W])$"
    lon: float
    lat: float

    def get(self) -> Navaid:  # type: ignore
        return Navaid(
            cast(str, self.elt),
            "NDB",
            self.lat,
            self.lon,
            float("nan"),
            None,
            None,
            None,
        )

    def __init__(self, elt: str) -> None:
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

        self.elt = elt  # type: ignore


class SpeedLevelChangePoint(_ElementaryBlock):
    pattern = "(.*)/(.*)"

    def get(self) -> Optional[Navaid]:  # type: ignore
        return self.elt[0].get()  # type: ignore

    @property
    def name(self) -> str:
        return self.elt[0].name  # type: ignore

    @property
    def text(self) -> str:
        return f"{self.elt[0].name} ({self.elt[1].name})"  # type: ignore


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

    @lru_cache()
    def _repr_svg_(self) -> Optional[str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return super()._repr_svg_()
            except Exception:
                return None

    def leaflet(self, **kwargs: Any) -> "Optional[LeafletPolyline]":
        raise ImportError(
            "Install ipyleaflet or traffic with the leaflet extension"
        )

    def skyvector(self) -> Dict[str, Any]:
        from ..data import client

        c = client.get(
            "https://skyvector.com/api/routes?dep={}&dst={}".format(
                self.origin, self.destination
            )
        )
        c.raise_for_status()

        c = client.get(
            f"https://skyvector.com/api/fpl?cmd=route&route={self.repr}"
        )
        c.raise_for_status()

        return c.json()  # type: ignore

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
        return linemerge([x.shape for x in self._parse() if x is not None])

    @lru_cache()
    def _parse(self) -> List[Any]:
        cumul: List[Union[None, ShapelyMixin]] = list()
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

            if isinstance(e, DirectPoint):
                from traffic.data import navaids

                previous, next_ = elts[i - 1], elts[i]

                if previous is None or next_ is None:
                    warnings.warn(f"Missing information around {elts[i]}")
                    continue

                if len(cumul) > 0 and cumul[-1] is not None:
                    # avoid obvious duplicates
                    elt1, *_, elt2 = cumul[-1].shape.coords
                    lon1, lat1, *_ = elt1
                    lon2, lat2, *_ = elt2
                    lon1, lon2 = min(lon1, lon2), max(lon1, lon2)
                    lat1, lat2 = min(lat1, lat2), max(lat1, lat2)
                    buf = 10  # conservative
                    # this one may return None
                    # (probably no, but mypy is whining)
                    n = navaids.extent(
                        (
                            lon1 - buf,
                            lon2 + buf,
                            lat1 - buf,
                            lat2 + buf,
                        )
                    )
                elif self.destination is not None and self.origin is not None:
                    from traffic.data import airports

                    origin = airports[self.origin]
                    destination = airports[self.destination]
                    assert origin is not None and destination is not None
                    lat1, lon1 = origin.latlon
                    lat2, lon2 = destination.latlon

                    lat1, lat2 = min(lat1, lat2), max(lat1, lat2)
                    lon1, lon2 = min(lon1, lon2), max(lon1, lon2)
                    buf = 2
                    n = navaids.extent(
                        (
                            lon1 - buf,
                            lon2 + buf,
                            lat1 - buf,
                            lat2 + buf,
                        )
                    )
                else:
                    n = None

                if n is None:
                    n = navaids

                p1, p2 = n.get(previous.name), n.get(next_.name)
                if p1 is None or p2 is None:
                    warnings.warn(
                        f"Could not find {previous.name} or {next_.name}"
                    )
                    continue
                cumul.append(Route("DCT", [p1, p2]))
                continue

            if isinstance(e, Direct):
                from traffic.data import navaids

                # previous, next_ = elts[i - 1], elts[i + 1]
                # if flightplan ends with DCT, lets[i+1] will be out of bounds
                previous = elts[i - 1]
                next_ = None if i + 1 >= len(elts) else elts[i + 1]

                if previous is None or next_ is None:
                    warnings.warn(f"Missing information around {elts[i]}")
                    continue

                if len(cumul) > 0 and cumul[-1] is not None:
                    # avoid obvious duplicates
                    elt1, *_, elt2 = cumul[-1].shape.coords
                    lon1, lat1, *_ = elt1
                    lon2, lat2, *_ = elt2
                    lon1, lon2 = min(lon1, lon2), max(lon1, lon2)
                    lat1, lat2 = min(lat1, lat2), max(lat1, lat2)
                    buf = 10  # conservative
                    # this one may return None
                    # (probably no, but mypy is whining)
                    n = navaids.extent(
                        (
                            lon1 - buf,
                            lon2 + buf,
                            lat1 - buf,
                            lat2 + buf,
                        )
                    )
                elif self.destination is not None and self.origin is not None:
                    from traffic.data import airports

                    origin = airports[self.origin]
                    destination = airports[self.destination]
                    assert origin is not None and destination is not None

                    lat1, lon1 = origin.latlon
                    lat2, lon2 = destination.latlon

                    lat1, lat2 = min(lat1, lat2), max(lat1, lat2)
                    lon1, lon2 = min(lon1, lon2), max(lon1, lon2)
                    buf = 2
                    n = navaids.extent(
                        (
                            lon1 - buf,
                            lon2 + buf,
                            lat1 - buf,
                            lat2 + buf,
                        )
                    )
                else:
                    n = None

                if n is None:
                    n = navaids

                p1, p2 = n.get(previous.name), n.get(next_.name)
                if p1 is None or p2 is None:
                    warnings.warn(
                        f"Could not find {previous.name} or {next_.name}"
                    )
                    continue
                cumul.append(Route("DCT", [p1, p2]))

        return cumul

    def _points(self, all_points: bool = False) -> Dict[str, PointBase]:
        cumul = dict()

        for elt in self._parse():
            if elt is not None:
                first, *args, last = elt.navaids
                cumul[first.name] = PointBase(
                    first.latitude, first.longitude, float("nan"), first.name
                )

                if all_points:
                    for elt in args:
                        cumul[elt.name] = PointBase(
                            elt.latitude, elt.longitude, float("nan"), elt.name
                        )
                cumul[last.name] = PointBase(
                    last.latitude, last.longitude, float("nan"), last.name
                )

        return cumul

    @property
    def points(self) -> List[PointBase]:
        return list(self._points().values())

    @property
    def all_points(self) -> List[PointBase]:
        return list(self._points(all_points=True).values())

    # -- Visualisation --
    def plot(
        self,
        ax: "GeoAxes",
        airports: bool = True,
        airports_kw: Optional[Dict[str, Any]] = None,
        labels: Union[None, bool, str] = None,
        labels_kw: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List["Artist"]:  # coverage: ignore
        """Plots the trajectory on a Matplotlib axis.

        FlightPlans support Cartopy axis as well with automatic projection. If
        no projection is provided, a default `PlateCarree
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html#platecarree>`_
        is applied.

        Example usage:

        .. code:: python

            from cartes.crs import Mercator
            fig, ax = plt.subplots(1, subplot_kw=dict(projection=Mercator())
            flightplan.plot(ax, labels=True, alpha=.5)

        .. note::
            See also `geoencode() <#traffic.core.Flight.geoencode>`_ for the
            altair equivalent.

        """

        from cartopy.crs import PlateCarree

        from ..visualize import markers

        cumul = []
        if "projection" in ax.__dict__ and "transform" not in kwargs:
            kwargs["transform"] = PlateCarree()

        if self.shape is not None:
            if isinstance(self.shape, LineString):
                cumul.append(ax.plot(*self.shape.xy, **kwargs))
            else:
                for s_ in self.shape:
                    cumul.append(ax.plot(*s_.xy, **kwargs))

        airports_style = dict(s=50, marker=markers.atc_tower)
        if airports_kw is not None:
            airports_style = {**airports_style, **airports_kw}

        labels_style = dict(s=30, marker="^", zorder=3)
        if labels_kw is not None:
            labels_style = {**labels_style, **labels_kw}

        if airports and self.origin:
            from traffic.data import airports as airport_db

            ap = airport_db[self.origin]
            if ap is not None:
                cumul.append(
                    ap.point.plot(ax, **airports_style)  # type: ignore
                )
        if airports and self.destination:
            from traffic.data import airports as airport_db

            ap = airport_db[self.destination]
            if ap is not None:
                cumul.append(
                    ap.point.plot(ax, **airports_style)  # type: ignore
                )

        if labels:
            for point in self.all_points if labels == "all" else self.points:
                cumul.append(point.plot(ax, **labels_style))  # type: ignore

        return cumul


def patch_leaflet() -> None:
    from ..visualize.leaflet import flightplan_leaflet

    FlightPlan.leaflet = flightplan_leaflet  # type: ignore


try:
    patch_leaflet()
except Exception:
    pass
