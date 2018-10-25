from datetime import datetime, timedelta
from typing import Optional, Set, Tuple, Union

from matplotlib.patches import Polygon as MplPolygon

import pandas as pd
import requests
from cartopy.crs import PlateCarree
from cartopy.mpl.geoaxes import GeoAxesSubplot
from shapely.geometry import Polygon

from ...core import Flight
from ...core.mixins import PointMixin, ShapelyMixin
from ...core.time import timelike, to_datetime
from ..basic.airport import Airport
from .opensky_impala import Impala


class Coverage:
    def __init__(self, json):
        self.df = pd.DataFrame.from_records(
            [
                dict(latitude=latitude, longitude=longitude, altitude=altitude)
                for latitude, longitude, altitude in json
            ]
        ).sort_values("altitude")

    def plot(self, ax, cmap="inferno", s=5, **kwargs):
        ax.scatter(
            self.df.longitude,
            self.df.latitude,
            s=s,
            transform=PlateCarree(),
            c=-self.df.altitude,
            cmap=cmap,
            **kwargs,
        )


class SensorRange(ShapelyMixin):
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

    def plot(self, ax: GeoAxesSubplot, **kwargs) -> None:

        if "facecolor" not in kwargs:
            kwargs["facecolor"] = "None"
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = ax._get_lines.get_next_color()

        if "projection" in ax.__dict__:
            ax.add_geometries([self.shape], crs=PlateCarree(), **kwargs)
        else:
            ax.add_patch(MplPolygon(list(self.shape.exterior.coords), **kwargs))


class OpenSky(Impala):

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

    def online_aircraft(self, own=False) -> pd.DataFrame:
        what = (
            "own"
            if (own and self.username != "" and self.password != "")
            else "all"
        )
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
        return self._format_dataframe(r, nautical_units=True)

    def online_track(self, icao24: str) -> Flight:
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
        ).assign(
            icao24=json["icao24"],
            callsign=json["callsign"],
        )
        return Flight(self._format_dataframe(df, nautical_units=True))

    def get_route(self, callsign: str) -> Tuple[Airport, ...]:
        c = requests.get(
            f"https://opensky-network.org/api/routes?callsign={callsign}"
        )
        if c.status_code == 404:
            raise ValueError("Unknown callsign")
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        json = c.json()
        from traffic.data import airports

        return tuple(airports[a] for a in json["route"])

    def get_aircraft(
        self,
        icao24: str,
        begin: Optional[timelike] = None,
        end: Optional[timelike] = None,
    ) -> pd.DataFrame:
        if begin is None:
            begin = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
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

    def sensor_list(self) -> pd.DataFrame:
        c = requests.get("https://opensky-network.org/api/sensor/list")
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return (
            pd.DataFrame.from_records(
                list(
                    {
                        **dict(
                            (
                                (key, value)
                                for (key, value) in x.items()
                                if key != "position"
                            )
                        ),
                        **x["position"],
                    }
                    for x in c.json()
                )
            )
            .assign(
                added=lambda df: df.added.apply(datetime.fromtimestamp),
                lastConnectionEvent=lambda df: df.lastConnectionEvent.apply(
                    datetime.fromtimestamp
                ),
            )
            .rename(columns=dict(lastConnectionEvent="last"))
            .drop(
                columns=[
                    "address",
                    "anonymized",
                    "anonymousPosition",
                    "hostname",
                    "deleted",
                    "id",
                    "notes",
                    "operator",
                    "port",
                ]
            )
        )

    @property
    def my_sensors(self) -> Set[str]:
        today = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        c = requests.get(
            f"https://opensky-network.org/api/sensor/myStats"
            f"?days={int(today.timestamp())}",
            auth=self.auth,
        )
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return set(c.json()[0]["stats"].keys())

    def sensor_range(
        self, serial: str, date: Optional[timelike] = None
    ) -> SensorRange:
        if date is None:
            date = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
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

    def global_coverage(self) -> Coverage:
        c = requests.get("https://opensky-network.org/api/range/coverage")
        if c.status_code != 200:
            raise ValueError(c.content.decode())
        return Coverage(c.json())

    def arrival(
        self,
        airport: Union[str, Airport],
        begin: Optional[timelike] = None,
        end: Optional[timelike] = None,
    ) -> pd.DataFrame:
        if begin is None:
            begin = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        begin = to_datetime(begin)
        if end is None:
            end = begin + timedelta(days=1)
        else:
            end = to_datetime(end)

        begin = int(begin.timestamp())
        end = int(end.timestamp())

        c = requests.get(
            f"https://opensky-network.org/api/flights/arrival"
            f"?begin={begin}&airport={airport}&end={end}"
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

    def departure(
        self,
        airport: Union[str, Airport],
        begin: Optional[timelike] = None,
        end: Optional[timelike] = None,
    ) -> pd.DataFrame:
        if begin is None:
            begin = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        begin = to_datetime(begin)
        if end is None:
            end = begin + timedelta(days=1)
        else:
            end = to_datetime(end)

        begin = int(begin.timestamp())
        end = int(end.timestamp())

        c = requests.get(
            f"https://opensky-network.org/api/flights/departure"
            f"?begin={begin}&airport={airport}&end={end}"
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
