from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.mixins import GeoDBMixin
from ...core.structure import Navaid, Route

_log = logging.getLogger(__name__)


class Airways(GeoDBMixin):
    """
    An ATS route is a specified route designed for channelling the flow of
    traffic as necessary for the provision of air traffic services.

    The term “ATS route” is used to mean variously, airway, advisory route,
    controlled or uncontrolled route, arrival or departure route, etc.

    An ATS route is defined by route specifications which include an ATS route
    designator, the track to or from significant points (waypoints), distance
    between significant points, reporting requirements and, as determined by the
    appropriate ATS authority, the lowest safe altitude. (ICAO Annex 11 - Air
    Traffic Services)

    A (deprecated) database of world ATS routes is available as:

    >>> from traffic.data import airways


    Any ATS route can be accessed by the bracket notation:

    >>> airways['Z50']
    Route('Z50', navaids=['EGOBA', 'SOT', 'BULTI', 'AYE', 'AVMON', ...])


    >>> airways.extent((-0.33, 4.85, 42.34, 45.05))["UN869"]
    Route('UN869', navaids=['XOMBO', 'TIVLI', 'AGN', 'NARAK', 'NASEP', ...])


    !!! note
        The following snippet plots the (in)famous `Silk Road Airway (L888)
        <https://flugdienstberater.org/l888>`_ over the Himalaya mountains,
        which requires special qualifications.

    .. jupyter-execute::

        import matplotlib.pyplot as plt

        from traffic.data import airways
        from cartes.crs import Orthographic

        with plt.style.context("traffic"):

            fig, ax = plt.subplots(
                figsize=(7, 7),
                subplot_kw=dict(projection=Orthographic(95, 30)),
            )

            ax.stock_img()
            ax.coastlines()

            airways["L888"].plot(
                ax, linewidth=2, linestyle="solid", color="crimson"
                )

            for navaid in airways["L888"].navaids:
                navaid.plot(
                    ax, s=20, marker=".", color="crimson",
                    text_kw=dict(fontsize=8)
                )

    """

    cache_path: Path

    def __init__(self, data: None | pd.DataFrame = None) -> None:
        self._data = data
        self._resolver_source: None | str = None

    def source(self, source: str) -> "Airways":
        clone = self.__class__(self._data)
        clone._extent = self._extent
        clone._resolver_source = source
        return clone

    def parse_data(self) -> pd.DataFrame:  # coverage: ignore
        cache_file = Path(__file__).parent.parent / "navdata" / "earth_awy.dat"
        assert cache_file.exists()
        self._data = pd.read_csv(cache_file, sep=" ", header=None)
        self._data.columns = ["route", "id", "navaid", "latitude", "longitude"]
        self._data.to_parquet(self.cache_path / "traffic_airways.parquet")
        assert self._data is not None
        return self._data

    @property
    def available(self) -> bool:
        return True

    @property
    def data(self) -> pd.DataFrame:
        if self._resolver_source is not None:
            from .. import resolver

            frame = resolver.data(source=self._resolver_source, kind="airway")
            if frame.empty:
                raise RuntimeError(
                    f"No airway data found for source {self._resolver_source}"
                )

            if {"name", "points"}.issubset(frame.columns):
                rows = []
                for _, row in frame.iterrows():
                    route_name = row["name"]
                    points = (
                        row["points"] if isinstance(row["points"], list) else []
                    )
                    for idx, point in enumerate(points):
                        rows.append(
                            {
                                "route": route_name,
                                "id": idx,
                                "navaid": point.get("code"),
                                "latitude": point.get("latitude", 0.0),
                                "longitude": point.get("longitude", 0.0),
                            }
                        )
                return pd.DataFrame.from_records(rows)

            return frame

        if self._data is not None:
            return self._data

        if not (self.cache_path / "traffic_airways.parquet").exists():
            self.parse_data()
        else:
            _log.info("Loading airways database")
            self._data = pd.read_parquet(
                self.cache_path / "traffic_airways.parquet"
            )

        if self._data is not None:
            self._data = self._data.rename(
                columns=dict(lat="latitude", lon="longitude")
            )

        assert self._data is not None
        return self._data

    def __getitem__(self, key: str) -> Route:
        return self.get(key)

    def global_get(self, name: str) -> Route:
        _log.warning("Use .get() function instead", DeprecationWarning)
        return self.get(name)

    def get(
        self,
        name: str,
        source: None | str = None,
        **kwargs: Any,
    ) -> Route:
        selected_source = (
            source if source is not None else self._resolver_source
        )

        if selected_source is None and self._data is not None:
            route = self._data[
                self._data["route"].str.upper() == name.upper()
            ].sort_values("id")
            if route.empty:
                raise AttributeError(f"Route {name} not found")
            return Route(
                name.upper(),
                [
                    Navaid(
                        str(row["navaid"]),
                        "FIX",
                        float(row["latitude"]),
                        float(row["longitude"]),
                        0,
                        None,
                        None,
                        None,
                    )
                    for _, row in route.iterrows()
                ],
            )

        from .. import resolver

        result = resolver.resolve(airway=name, source=selected_source, **kwargs)
        if result.selected is None:
            if source is None:
                raise AttributeError(f"Route {name} not found")
            raise AttributeError(f"Route {name} not found in source {source}")

        points = result.selected.payload.get("points", [])
        navaids = [
            Navaid(
                str(point.get("code") or ""),
                str(point.get("kind") or "FIX").upper(),
                float(point.get("latitude") or 0.0),
                float(point.get("longitude") or 0.0),
                0,
                None,
                None,
                None,
            )
            for point in points
        ]
        return Route(result.selected.code, navaids)

    def search(self, name: str) -> "Airways":
        """
        Selects the subset of airways matching name in the route name or in the
        passed navigational beacon.

        >>> airways.extent('Switzerland').search("Z50")  # doctest: +SKIP
          route   id   navaid   latitude   longitude
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Z50     7    GERSA    47.04      8.532
          Z50     8    KELIP    46.96      8.762
          Z50     9    SOPER    46.89      8.944
          Z50     10   PELAD    46.6       9.726
          Z50     11   RESIA    46.48      10.04


        >>> airways.search("NARAK")  # doctest: +SKIP
          route   id   navaid   latitude   longitude
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          N869    88   NARAK    44.3       1.749
          UN859   15   NARAK    44.3       1.749
          UN869   23   NARAK    44.3       1.749
          UT122   15   NARAK    44.3       1.749
          UY155   2    NARAK    44.3       1.749
          UZ365   3    NARAK    44.3       1.749

        """
        output = self.__class__(
            self.data.query("route == @name.upper() or navaid == @name.upper()")
        )
        return output
