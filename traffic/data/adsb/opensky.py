from datetime import datetime, timedelta

import pandas as pd
import requests
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.artist import Artist
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from typing import Optional, Set, Tuple, Union

from ...core import Flight, StateVectors as SVMixin
from ...core.mixins import PointMixin, ShapelyMixin
from ...core.time import round_time, timelike, to_datetime
from ..basic.airport import Airport
from .opensky_impala import Impala


class Coverage(object):
    """Plots the output of the coverage json."""

    def __init__(self, json):
        self.df = pd.DataFrame.from_records(
            [
                dict(latitude=latitude, longitude=longitude, altitude=altitude)
                for latitude, longitude, altitude in json
            ]
        ).sort_values("altitude")

    def plot(
        self, ax: GeoAxesSubplot, cmap: str = "inferno", s: int = 5, **kwargs
    ) -> Artist:
        """Plotting function. All arguments are passed to ax.scatter"""
        return ax.scatter(
            self.df.longitude,
            self.df.latitude,
            s=s,
            transform=PlateCarree(),
            c=-self.df.altitude,
            cmap=cmap,
            **kwargs,
        )


class StateVectors(SVMixin):
    """Plots the state vectors returned by OpenSky REST API."""

    def __init__(self, data: pd.DataFrame, opensky: "OpenSky") -> None:
        super().__init__(data)
        self.opensky = opensky

    def __getitem__(self, identifier: str):
        icao24 = self.data.query(
            "callsign == @identifier or icao24 == @identifier"
        ).icao24.item()
        return self.opensky.api_tracks(icao24)


class SensorRange(ShapelyMixin):
    """Wraps the polygon defining the range of an OpenSky sensor."""

    def __init__(self, json):
        for key, value in json.items():
            self.shape = Polygon(
                [(lon, lat) for (deg, lat, lon) in value[0]["ranges"]]
            )
            self.point = PointMixin()

            self.point.latitude, self.point.longitude = value[0][
                "sensorPosition"
            ]
            self.point.name = value[0]["serial"]

    def plot(self, ax: GeoAxesSubplot, **kwargs) -> Artist:
        """Plotting function. All arguments are passed to the geometry"""

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


class OpenSky(Impala):
    """Wrapper to OpenSky REST API and Impala Shell.

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

    # All Impala specific functions are implemented in opensky_impala.py

    _json_columns = [
        "icao24",
        "callsign",
        "origin_country",
        "last_position",
        "timestamp",
        "longitude",
        "latitude",
        "altitude",
        "onground",
        "groundspeed",
        "track",
        "vertical_rate",
        "sensors",
        "geoaltitude",
        "squawk",
        "spi",
        "position_source",
    ]

    def api_states(
        self,
        own: bool = False,
        bounds: Union[
            BaseGeometry, Tuple[float, float, float, float], None
        ] = None,
    ) -> StateVectors:
        """Returns the current state vectors from OpenSky REST API.

        If own parameter is set to True, returns only the state vectors
        associated to own sensors (requires authentication)

        bounds parameter can be a shape or a tuple of float.

        Official documentation
        ----------------------

        Limitiations for anonymous (unauthenticated) users

        Anonymous are those users who access the API without using credentials.
        The limitations for anonymous users are:

        Anonymous users can only get the most recent state vectors, i.e. the
        time parameter will be ignored.  Anonymous users can only retrieve data
        with a time resultion of 10 seconds. That means, the API will return
        state vectors for time now − (now mod 10)

        Limitations for OpenSky users

        An OpenSky user is anybody who uses a valid OpenSky account (see below)
        to access the API. The rate limitations for OpenSky users are:

        - OpenSky users can retrieve data of up to 1 hour in the past. If the
        time parameter has a value t < now−3600 the API will return
        400 Bad Request.

        - OpenSky users can retrieve data with a time resultion of 5 seconds.
        That means, if the time parameter was set to t , the API will return
        state vectors for time t−(t mod 5).

        """

        what = "own" if (own and self.auth is not None) else "all"

        if bounds is not None:
            try:
                # thinking of shapely bounds attribute (in this order)
                # I just don't want to add the shapely dependency here
                west, south, east, north = bounds.bounds  # type: ignore
            except AttributeError:
                west, south, east, north = bounds

            what += f"?lamin={south}&lamax={north}&lomin={west}&lomax={east}"

        c = requests.get(
            f"https://opensky-network.org/api/states/{what}", auth=self.auth
        )

        if c.status_code != 200:
            raise ValueError(c.content.decode())
        r = pd.DataFrame.from_records(
            c.json()["states"], columns=self._json_columns
        )
        r = r.drop(["origin_country", "spi", "sensors"], axis=1)
        r = r.dropna()

        return StateVectors(
            self._format_dataframe(r, nautical_units=True), self
        )

    def api_tracks(self, icao24: str) -> Flight:
        """Returns a Flight corresponding to a given aircraft.

        Official documentation
        ----------------------

        Retrieve the trajectory for a certain aircraft at a given time. The
        trajectory is a list of waypoints containing position, barometric
        altitude, true track and an on-ground flag.

        In contrast to state vectors, trajectories do not contain all
        information we have about the flight, but rather show the aircraft’s
        general movement pattern. For this reason, waypoints are selected among
        available state vectors given the following set of rules:

        - The first point is set immediately after the the aircraft’s expected
        departure, or after the network received the first poisition when the
        aircraft entered its reception range.
        - The last point is set right before the aircraft’s expected arrival, or
        the aircraft left the networks reception range.
        - There is a waypoint at least every 15 minutes when the aircraft is
        in-flight.
        - A waypoint is added if the aircraft changes its track more than 2.5°.
        - A waypoint is added if the aircraft changes altitude by more than 100m
        (~330ft).
        - A waypoint is added if the on-ground state changes.

        Tracks are strongly related to flights. Internally, we compute flights
        and tracks within the same processing step. As such, it may be
        benificial to retrieve a list of flights with the API methods from
        above, and use these results with the given time stamps to retrieve
        detailed track information.

        """
        c = requests.get(
            f"https://opensky-network.org/api/tracks/?icao24={icao24}"
        )
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        json = c.json()

        df = pd.DataFrame.from_records(
            json["path"],
            columns=[
                "timestamp",
                "latitude",
                "longitude",
                "altitude",
                "track",
                "onground",
            ],
        ).assign(icao24=json["icao24"], callsign=json["callsign"])
        return Flight(self._format_dataframe(df, nautical_units=True))

    def api_routes(self, callsign: str) -> Tuple[Airport, ...]:
        """Returns the route associated to a callsign."""
        from .. import airports

        c = requests.get(
            f"https://opensky-network.org/api/routes?callsign={callsign}"
        )
        if c.status_code == 404:
            raise ValueError("Unknown callsign")
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        json = c.json()

        return tuple(airports[a] for a in json["route"])

    def api_aircraft(
        self,
        icao24: str,
        begin: Optional[timelike] = None,
        end: Optional[timelike] = None,
    ) -> pd.DataFrame:
        """Returns a flight table associated to an aircraft.

        Official documentation
        ----------------------

        This API call retrieves flights for a particular aircraft within a
        certain time interval. Resulting flights departed and arrived within
        [begin, end]. If no flights are found for the given period, HTTP stats
        404 - Not found is returned with an empty response body.

        """

        if begin is None:
            begin = round_time(datetime.now(), by=timedelta(days=1))
        begin = to_datetime(begin)
        if end is None:
            end = begin + timedelta(days=1)
        else:
            end = to_datetime(end)

        begin = int(begin.timestamp())
        end = int(end.timestamp())

        c = requests.get(
            f"https://opensky-network.org/api/flights/aircraft"
            f"?icao24={icao24}&begin={begin}&end={end}"
        )
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return (
            pd.DataFrame.from_records(c.json())[
                [
                    "firstSeen",
                    "lastSeen",
                    "icao24",
                    "callsign",
                    "estDepartureAirport",
                    "estArrivalAirport",
                ]
            ]
            .assign(
                firstSeen=lambda df: df.firstSeen.apply(datetime.fromtimestamp),
                lastSeen=lambda df: df.lastSeen.apply(datetime.fromtimestamp),
            )
            .sort_values("lastSeen")
        )

    @property
    def api_sensors(self) -> Set[str]:
        """The set of sensors serials you own (require authentication)."""
        today = round_time(datetime.now(), by=timedelta(days=1))
        c = requests.get(
            f"https://opensky-network.org/api/sensor/myStats"
            f"?days={int(today.timestamp())}",
            auth=self.auth,
        )
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return set(c.json()[0]["stats"].keys())

    def api_range(
        self, serial: str, date: Optional[timelike] = None
    ) -> SensorRange:
        """Wraps a polygon representing a sensor's range.

        By default, returns the current range. Otherwise, you may enter a
        specific day (as a string, as an epoch or as a datetime)
        """

        if date is None:
            date = round_time(datetime.now(), by=timedelta(days=1))
        else:
            date = to_datetime(date)
        date = int(date.timestamp())

        c = requests.get(
            f"https://opensky-network.org/api/range/days"
            f"?days={date}&serials={serial}"
        )
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return SensorRange(c.json())

    def api_global_coverage(self) -> Coverage:
        c = requests.get("https://opensky-network.org/api/range/coverage")
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return Coverage(c.json())

    def api_arrival(
        self,
        airport: Union[str, Airport],
        begin: Optional[timelike] = None,
        end: Optional[timelike] = None,
    ) -> pd.DataFrame:
        """Returns a flight table associated to an airport.

        By default, returns the current table. Otherwise, you may enter a
        specific date (as a string, as an epoch or as a datetime)

        Official documentation
        ----------------------

        Retrieve flights for a certain airport which arrived within a given time
        interval [begin, end]. If no flights are found for the given period,
        HTTP stats 404 - Not found is returned with an empty response body.

        """

        if isinstance(airport, str):
            from .. import airports

            airport_code = airports[airport].icao
        else:
            airport_code = airport.icao

        if begin is None:
            begin = round_time(datetime.now(), by=timedelta(days=1))
        begin = to_datetime(begin)
        if end is None:
            end = begin + timedelta(days=1)
        else:
            end = to_datetime(end)

        begin = int(begin.timestamp())
        end = int(end.timestamp())

        c = requests.get(
            f"https://opensky-network.org/api/flights/arrival"
            f"?begin={begin}&airport={airport_code}&end={end}"
        )

        if c.status_code != 200:
            raise ValueError(c.content.decode())

        return (
            pd.DataFrame.from_records(c.json())[
                [
                    "firstSeen",
                    "lastSeen",
                    "icao24",
                    "callsign",
                    "estDepartureAirport",
                    "estArrivalAirport",
                ]
            ]
            .assign(
                firstSeen=lambda df: df.firstSeen.apply(datetime.fromtimestamp),
                lastSeen=lambda df: df.lastSeen.apply(datetime.fromtimestamp),
            )
            .sort_values("lastSeen")
        )

    def api_departure(
        self,
        airport: Union[str, Airport],
        begin: Optional[timelike] = None,
        end: Optional[timelike] = None,
    ) -> pd.DataFrame:
        """Returns a flight table associated to an airport.

        By default, returns the current table. Otherwise, you may enter a
        specific date (as a string, as an epoch or as a datetime)

        Official documentation
        ----------------------

        Retrieve flights for a certain airport which departed within a given
        time interval [begin, end]. If no flights are found for the given
        period, HTTP stats 404 - Not found is returned with an empty response
        body.

        """

        if isinstance(airport, str):
            from .. import airports

            airport_code = airports[airport].icao
        else:
            airport_code = airport.icao

        if begin is None:
            begin = round_time(datetime.now(), by=timedelta(days=1))
        begin = to_datetime(begin)
        if end is None:
            end = begin + timedelta(days=1)
        else:
            end = to_datetime(end)

        begin = int(begin.timestamp())
        end = int(end.timestamp())

        c = requests.get(
            f"https://opensky-network.org/api/flights/departure"
            f"?begin={begin}&airport={airport_code}&end={end}"
        )

        if c.status_code != 200:
            raise ValueError(c.content.decode())

        return (
            pd.DataFrame.from_records(c.json())[
                [
                    "firstSeen",
                    "lastSeen",
                    "icao24",
                    "callsign",
                    "estDepartureAirport",
                    "estArrivalAirport",
                ]
            ]
            .assign(
                firstSeen=lambda df: df.firstSeen.apply(datetime.fromtimestamp),
                lastSeen=lambda df: df.lastSeen.apply(datetime.fromtimestamp),
            )
            .sort_values("firstSeen")
        )
