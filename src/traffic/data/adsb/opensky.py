from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Tuple,
    TypedDict,
    TypeVar,
    overload,
)

from pyopensky import impala, rest, schema, trino
from pyopensky.api import OpenSkyDBAPI

import pandas as pd
from shapely.geometry import Polygon

from ...core import Flight, StateVectors, Traffic
from ...core.mixins import PointMixin, ShapelyMixin
from ...core.time import timelike
from ...core.types import HasBounds, ProgressbarType
from ..basic.airports import Airport
from .decode import RawData
from .flarm import FlarmData

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxesSubplot
    from matplotlib.artist import Artist

_log = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class Coverage(object):
    """Plots the output of the coverage json."""

    def __init__(self, json: List[Tuple[float, float, float]]) -> None:
        self.df = pd.DataFrame.from_records(
            [
                dict(latitude=latitude, longitude=longitude, altitude=altitude)
                for latitude, longitude, altitude in json
            ]
        ).sort_values("altitude")

    def plot(
        self,
        ax: "GeoAxesSubplot",
        cmap: str = "inferno",
        s: int = 5,
        **kwargs: Any,
    ) -> "Artist":
        """Plotting function. All arguments are passed to ax.scatter"""
        from cartopy.crs import PlateCarree

        return ax.scatter(
            self.df.longitude,
            self.df.latitude,
            s=s,
            transform=PlateCarree(),
            c=-self.df.altitude,
            cmap=cmap,
            **kwargs,
        )


class SensorRangeJSON(TypedDict):
    serial: str
    ranges: List[Tuple[float, float, float]]
    sensorPosition: Tuple[float, float]


class SensorRange(ShapelyMixin):
    """Wraps the polygon defining the range of an OpenSky sensor."""

    def __init__(self, json: Dict[Any, List[SensorRangeJSON]]) -> None:
        for _, value in json.items():
            self.shape = Polygon(
                [(lon, lat) for (_deg, lat, lon) in value[0]["ranges"]]
            )
            self.point = PointMixin()

            self.point.latitude, self.point.longitude = value[0][
                "sensorPosition"
            ]
            self.point.name = value[0]["serial"]

    def plot(self, ax: "GeoAxesSubplot", **kwargs: Any) -> "Artist":
        """Plotting function. All arguments are passed to the geometry"""
        from cartopy.crs import PlateCarree
        from matplotlib.patches import Polygon as MplPolygon

        if "facecolor" not in kwargs:
            kwargs["facecolor"] = "None"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = ax._get_lines.get_next_color()

        if "projection" in ax.__dict__:
            return ax.add_geometries([self.shape], crs=PlateCarree(), **kwargs)
        else:
            return ax.add_patch(
                MplPolygon(list(self.shape.exterior.coords), **kwargs)
            )


def copy_documentation(source_fun: Any) -> Callable[[F], F]:
    def new_function(input_fun: F) -> F:
        input_fun.__doc__ = source_fun.__doc__
        return input_fun

    return new_function


def format_history(
    df: pd.DataFrame, nautical_units: bool = True
) -> pd.DataFrame:
    """
    This function can be used in tandem with `_format_dataframe()` to
    convert (historical data specific) column types and optionally convert
    the units back to nautical miles, feet and feet/min.

    """

    if "lastcontact" in df.columns:
        df = df.drop(["lastcontact"], axis=1)

    if "lat" in df.columns and df.lat.dtype == object:
        df = df[df.lat != "lat"]  # header is regularly repeated

    # restore all types
    for column_name in [
        "lat",
        "lon",
        "velocity",
        "heading",
        "vertrate",
        "baroaltitude",
        "geoaltitude",
        # "lastposupdate",
        # "lastcontact",
    ]:
        if column_name in df.columns:
            df[column_name] = df[column_name].astype(float)

    if "onground" in df.columns and df.onground.dtype != bool:
        df.onground = df.onground == "true"
        df.alert = df.alert == "true"
        df.spi = df.spi == "true"

    # better (to me) formalism about columns
    df = df.rename(
        columns={
            "lat": "latitude",
            "lon": "longitude",
            "heading": "track",
            "velocity": "groundspeed",
            "vertrate": "vertical_rate",
            "baroaltitude": "altitude",
            "time": "timestamp",
            "lastposupdate": "last_position",
        }
    )

    if nautical_units:
        df.altitude = (df.altitude / 0.3048).round(0)
        if "geoaltitude" in df.columns:
            df.geoaltitude = (df.geoaltitude / 0.3048).round(0)
        if "groundspeed" in df.columns:
            df.groundspeed = (df.groundspeed / 1852 * 3600).round(0)
        if "vertical_rate" in df.columns:
            df.vertical_rate = (df.vertical_rate / 0.3048 * 60).round(0)

    return df


class OpenSky:
    """Wrapper to OpenSky REST API, Trino database and Impala Shell.

    An instance is automatically constructed when importing traffic.data with
    the name opensky. Credentials are fetched from the configuration file.

    [global]
    opensky_username =
    opensky_password =

    The configuration file is located at `traffic.config_file`.

    All functions from the REST API are prefixed with `api_`. The other
    functions wrap the access to the Impala shell.

    REST API is documented here: https://opensky-network.org/apidoc/rest.html

    """

    def __init__(self) -> None:
        self.rest_client = rest.REST()
        self.impala_client = impala.Impala()
        self.trino_client = trino.Trino()

        self.db_client: OpenSkyDBAPI = (
            self.trino_client
            if self.trino_client.token() is not None
            else self.impala_client
        )

    @copy_documentation(rest.REST.states)
    def api_states(
        self,
        own: bool = False,
        bounds: None
        | str
        | HasBounds
        | tuple[float, float, float, float] = None,
    ) -> StateVectors:
        df = self.rest_client.states(own, bounds).pipe(format_history)
        return StateVectors(df)

    @copy_documentation(rest.REST.tracks)
    def api_tracks(self, icao24: str, time: None | timelike = None) -> Flight:
        df = self.rest_client.tracks(icao24, time).pipe(format_history)
        return Flight(df)

    @copy_documentation(rest.REST.routes)
    def api_routes(self, callsign: str) -> tuple[str, str]:
        return self.rest_client.routes(callsign)

    @copy_documentation(rest.REST.aircraft)
    def api_aircraft(
        self,
        icao24: str,
        begin: None | timelike = None,
        end: None | timelike = None,
    ) -> pd.DataFrame:
        return self.rest_client.aircraft(icao24, begin, end)

    @copy_documentation(rest.REST.sensors)
    def api_sensors(self, day: None | timelike = None) -> set[str]:
        return self.rest_client.sensors(day)

    @copy_documentation(rest.REST.range)
    def api_range(
        self, serial: str, day: None | timelike = None
    ) -> SensorRange:
        json = self.rest_client.range(serial, day)
        return SensorRange(json)

    @copy_documentation(rest.REST.global_coverage)
    def api_global_coverage(self, day: None | timelike = None) -> Coverage:
        json = self.rest_client.global_coverage(day)
        return Coverage(json)

    @copy_documentation(rest.REST.arrival)
    def api_arrival(
        self,
        airport: str | Airport,
        begin: None | timelike = None,
        end: None | timelike = None,
    ) -> pd.DataFrame:
        return self.rest_client.arrival(
            airport if isinstance(airport, str) else airport.icao, begin, end
        )

    @copy_documentation(rest.REST.departure)
    def api_departure(
        self,
        airport: str | Airport,
        begin: None | timelike = None,
        end: None | timelike = None,
    ) -> pd.DataFrame:
        return self.rest_client.departure(
            airport if isinstance(airport, str) else airport.icao, begin, end
        )

    @copy_documentation(trino.Trino.flightlist)
    def flightlist(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,  # more reasonable to be explicit about arguments
        departure_airport: None | str | list[str] = None,
        arrival_airport: None | str | list[str] = None,
        airport: None | str | list[str] = None,
        callsign: None | str | list[str] = None,
        icao24: None | str | list[str] = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        **kwargs: Any,
    ) -> None | pd.DataFrame:
        return self.db_client.flightlist(
            start,
            stop,
            *args,
            departure_airport=departure_airport,
            arrival_airport=arrival_airport,
            airport=airport,
            callsign=callsign,
            icao24=icao24,
            cached=cached,
            compress=compress,
            limit=limit,
            **kwargs,
        )
        ...

    @overload
    def history(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,
        # date_delta: timedelta = timedelta(hours=1),
        callsign: None | str | list[str] = None,
        icao24: None | str | list[str] = None,
        serials: None | int | Iterable[int] = None,
        bounds: None
        | str
        | HasBounds
        | tuple[float, float, float, float] = None,
        departure_airport: None | str = None,
        arrival_airport: None | str = None,
        airport: None | str = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        return_flight: Literal[False] = False,
        **kwargs: Any,
    ) -> None | Traffic:
        ...

    @overload
    def history(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,
        # date_delta: timedelta = timedelta(hours=1),
        callsign: None | str | list[str] = None,
        icao24: None | str | list[str] = None,
        serials: None | int | Iterable[int] = None,
        bounds: None
        | str
        | HasBounds
        | tuple[float, float, float, float] = None,
        departure_airport: None | str = None,
        arrival_airport: None | str = None,
        airport: None | str = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        return_flight: Literal[True],
        **kwargs: Any,
    ) -> None | Flight:
        ...

    @copy_documentation(trino.Trino.history)
    def history(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,
        # date_delta: timedelta = timedelta(hours=1),
        callsign: None | str | list[str] = None,
        icao24: None | str | list[str] = None,
        serials: None | int | Iterable[int] = None,
        bounds: None
        | str
        | HasBounds
        | tuple[float, float, float, float] = None,
        departure_airport: None | str = None,
        arrival_airport: None | str = None,
        airport: None | str = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        return_flight: bool = False,
        **kwargs: Any,
    ) -> None | Flight | Traffic:
        df = self.db_client.history(
            start,
            stop,
            *args,
            callsign=callsign,
            icao24=icao24,
            serials=serials,
            bounds=bounds,
            departure_airport=departure_airport,
            arrival_airport=arrival_airport,
            airport=airport,
            cached=cached,
            compress=compress,
            limit=limit,
            **kwargs,
        )
        if df is None:
            return None

        df = format_history(df)

        if return_flight:
            return Flight(df)

        return Traffic(df)
        ...

    @copy_documentation(trino.Trino.rawdata)
    def rawdata(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,  # more reasonable to be explicit about arguments
        icao24: None | str | list[str] = None,
        serials: None | int | Iterable[int] = None,
        bounds: None | HasBounds | tuple[float, float, float, float] = None,
        callsign: None | str | list[str] = None,
        departure_airport: None | str = None,
        arrival_airport: None | str = None,
        airport: None | str = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        **kwargs: Any,
    ) -> None | RawData:
        df = self.db_client.rawdata(
            start,
            stop,
            *args,
            icao24=icao24,
            serials=serials,
            bounds=bounds,
            callsign=callsign,
            departure_airport=departure_airport,
            arrival_airport=arrival_airport,
            airport=airport,
            cached=cached,
            compress=compress,
            limit=limit,
            **kwargs,
        )

        if df is None:
            return None

        return RawData(df)

    @copy_documentation(trino.Trino.rawdata)
    def extended(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,  # more reasonable to be explicit about arguments
        icao24: None | str | list[str] = None,
        serials: None | int | Iterable[int] = None,
        bounds: None | HasBounds | tuple[float, float, float, float] = None,
        callsign: None | str | list[str] = None,
        departure_airport: None | str = None,
        arrival_airport: None | str = None,
        airport: None | str = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        **kwargs: Any,
    ) -> None | RawData:
        kwargs = {
            **kwargs,
            **(
                # this one is with impala
                dict(table_name="rollcall_replies_data4")
                if isinstance(self.db_client, impala.Impala)
                # this one is with Trino
                else dict(Table=schema.RollcallRepliesData4)  # default anyway
            ),
        }

        return self.rawdata(
            start,
            stop,
            *args,
            icao24=icao24,
            serials=serials,
            bounds=bounds,
            callsign=callsign,
            departure_airport=departure_airport,
            arrival_airport=arrival_airport,
            airport=airport,
            cached=cached,
            compress=compress,
            limit=limit,
            **kwargs,
        )

    @copy_documentation(impala.Impala.flarm)
    def flarm(
        self,
        start: timelike,
        stop: None | timelike = None,
        *args: Any,  # more reasonable to be explicit about arguments
        sensor_name: None | str | list[str] = None,
        cached: bool = True,
        compress: bool = False,
        limit: None | int = None,
        other_params: str = "",
        progressbar: bool | ProgressbarType[Any] = True,
    ) -> None | FlarmData:
        df = self.impala_client.flarm(
            start,
            stop,
            *args,
            sensor_name=sensor_name,
            cached=cached,
            compress=compress,
            limit=limit,
            other_params=other_params,
            progressbar=progressbar,
        )
        if df is None:
            return None

        else:
            return FlarmData(df)


# below this line is only helpful references
# ------------------------------------------
# [hadoop-1:21000] > describe rollcall_replies_data4;
# +----------------------+-------------------+---------+
# | name                 | type              | comment |
# +----------------------+-------------------+---------+
# | sensors              | array<struct<     |         |
# |                      |   serial:int,     |         |
# |                      |   mintime:double, |         |
# |                      |   maxtime:double  |         |
# |                      | >>                |         |
# | rawmsg               | string            |         |
# | mintime              | double            |         |
# | maxtime              | double            |         |
# | msgcount             | bigint            |         |
# | icao24               | string            |         |
# | message              | string            |         |
# | isid                 | boolean           |         |
# | flightstatus         | tinyint           |         |
# | downlinkrequest      | tinyint           |         |
# | utilitymsg           | tinyint           |         |
# | interrogatorid       | tinyint           |         |
# | identifierdesignator | tinyint           |         |
# | valuecode            | smallint          |         |
# | altitude             | double            |         |
# | identity             | string            |         |
# | hour                 | int               |         |
# +----------------------+-------------------+---------+
