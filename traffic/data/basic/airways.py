import re
from operator import itemgetter
from pathlib import Path
from typing import Optional, Set, Tuple, Union, cast

import pandas as pd
import requests
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

from .navaid import Navaid  # type: ignore
from ...core.mixins import ShapelyMixin

__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "ProfHoekstra/bluesky/master/data/navdata"

BoundsType = Union[BaseGeometry, Tuple[float, float, float, float]]


class Airway(ShapelyMixin):

    def __init__(self, shape):
        self.shape = shape

    def plot(self, ax, **kwargs):
        if "projection" in ax.__dict__:
            from cartopy.crs import PlateCarree
            kwargs["transform"] = PlateCarree()

        ax.plot(*self.shape.xy, **kwargs)


class Airways(object):

    cache: Optional[Path] = None

    def __init__(self, airways: Optional[pd.DataFrame] = None) -> None:
        if airways is None:
            if self.cache is not None and self.cache.exists():
                self.airways = pd.read_pickle(self.cache)
            else:
                self.initialize()
                # self.airways is set in above method
                self.airways = cast(pd.DataFrame, self.airways)
                if self.cache is not None:
                    self.airways.to_pickle(self.cache)
        else:
            self.airways = airways

    def initialize(self) -> None:
        c = requests.get(f"{base_url}/awy.dat")

        buffer = []
        for line in c.iter_lines():
            line = line.decode(encoding="ascii", errors="ignore").strip()
            if len(line) == 0 or line[0] == "#":
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            if not re.match("^[\d.]*$", fields[1]):
                continue

            for id_ in fields[-1].split("-"):
                buffer.append(
                    fields[:9]
                    + [id_]
                    + [
                        LineString(
                            [
                                [float(fields[2]), float(fields[1])],
                                [float(fields[5]), float(fields[4])],
                            ]
                        )
                    ]
                )

        airways = pd.DataFrame.from_records(
            buffer,
            columns=[
                "origin",
                "fromlat",
                "fromlon",
                "destination",
                "tolat",
                "tolon",
                "direction",
                "low",
                "up",
                "id",
                "linestring",
            ],
        )

        airways.low = airways.low.astype(int)
        airways.up = airways.up.astype(int)
        airways.fromlat = airways.fromlat.astype(float)
        airways.fromlon = airways.fromlon.astype(float)
        airways.tolat = airways.tolat.astype(float)
        airways.tolon = airways.tolon.astype(float)

        airways["bounds"] = airways.linestring.apply(lambda x: x.bounds)
        airways["west"] = airways.bounds.apply(itemgetter(0))
        airways["south"] = airways.bounds.apply(itemgetter(1))
        airways["east"] = airways.bounds.apply(itemgetter(2))
        airways["north"] = airways.bounds.apply(itemgetter(3))

        self.airways = airways

    def __getitem__(self, name: str) -> BaseGeometry:
        self.airways = cast(pd.DataFrame, self.airways)
        return Airway(
            linemerge(self.airways.groupby("id").get_group(name).linestring)
        )

    def through(
        self, navaid: Union[str, Navaid], min_upper: Optional[int] = None
    ) -> Optional[Set[str]]:

        if isinstance(navaid, str):
            from .. import navaids

            _navaid = navaids[navaid]
            if _navaid is None:
                return None
            navaid = _navaid

        navaid = cast(Navaid, navaid)
        self.airways = cast(pd.DataFrame, self.airways)

        subset = self.airways[
            (
                (self.airways.fromlat == navaid.lat)
                & (self.airways.fromlon == navaid.lon)
            )
            | (
                (self.airways.tolat == navaid.lat)
                & (self.airways.tolon == navaid.lon)
            )
        ]

        if min_upper is not None:
            subset = subset[subset.up >= min_upper]

        return set(subset.id)

    def intersects(
        self, bounds: BoundsType, min_upper: Optional[int] = None
    ) -> "Airways":

        if isinstance(bounds, BaseGeometry):
            bounds = bounds.bounds

        west, south, east, north = bounds

        self.airways = cast(pd.DataFrame, self.airways)

        candidates = self.airways[
            ((self.airways.west >= west) | (self.airways.east >= west))
            & ((self.airways.west <= east) | (self.airways.east <= east))
            & ((self.airways.south >= south) | (self.airways.north >= south))
            & ((self.airways.south <= north) | (self.airways.north <= north))
        ]

        if min_upper is not None:
            candidates = candidates[candidates.up >= min_upper]

        return Airways(candidates)

    def plot(self, ax, **kwargs):

        if "projection" in ax.__dict__:
            from cartopy.crs import PlateCarree

            kwargs["transform"] = PlateCarree()

        if "color" not in kwargs:
            kwargs["color"] = "#aaaaaa"

        if "linestyle" not in kwargs:
            kwargs["linestyle"] = ":"

        if "lw" not in kwargs:
            kwargs["lw"] = .5

        for line in self.airways.linestring:
            ax.plot(*line.xy, **kwargs)
