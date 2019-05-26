import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from ...core.mixins import DataFrameMixin, ShapelyMixin
from ...drawing import Nominatim, location

__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "xoolive/traffic/master/data/navdata"

BoundsType = Union[BaseGeometry, Tuple[float, float, float, float]]


class Route(ShapelyMixin):
    def __init__(self, shape: BaseGeometry, name: str, navaids: List[str]):
        self.shape = shape
        self.name = name
        self.navaids = navaids

    def __repr__(self):
        return f"{self.name} ({', '.join(self.navaids)})"

    def _info_html(self) -> str:
        title = f"<b>Route {self.name}</b><br/>"
        title += f"flies through {', '.join(self.navaids)}.<br/>"
        return title

    def _repr_html_(self) -> str:
        title = self._info_html()
        no_wrap_div = '<div style="white-space: nowrap">{}</div>'
        return title + no_wrap_div.format(self._repr_svg_())

    def plot(self, ax, **kwargs):  # coverage: ignore
        if "color" not in kwargs:
            kwargs["color"] = "#aaaaaa"
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5
        if "linewidth" not in kwargs and "lw" not in kwargs:
            kwargs["linewidth"] = 0.8
        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["linestyle"] = "dashed"
        if "projection" in ax.__dict__:
            from cartopy.crs import PlateCarree

            kwargs["transform"] = PlateCarree()

        ax.plot(*self.shape.xy, **kwargs)


class Airways(DataFrameMixin):
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

    >>> airways['UN869']
    UN869 (FTV, RUSIK, ADM, MABAP, PELAX, SLK, RBT, GALTO, PIMOS, MGA, BLN,
    ANZAN, NASOS, ADUXO, EDIMU, PISUS, EXEMU, ZAR, ELSAP, XOMBO, TIVLI, AGN,
    NARAK, NASEP, ROMAK, MINSO, MOKDI, LERGA, TITVA, REPSI, MEBAK, NINTU, MILPA,
    GVA10, VEROX, NEMOS, BENOT, LUTIX, OLBEN, RINLI, NATOR, TEDGO, GUPIN, DKB,
    ODEGU, AMOSA, KEGOS, ANELA, KEPOM, NOGRA, RONIG, OKG)

    .. note::
        The following snippet plots the (in)famous `Silk Road Airway (L888)
        <https://flugdienstberater.org/l888>`_ over the Himalaya mountains,
        which requires special qualifications.

    >>> from traffic.data import navaids
    >>> from traffic.drawing import Orthographic
    >>> with plt.style.context("traffic"):
    ...     fig, ax = plt.subplots(
    ...         1, figsize=(15, 15),
    ...         subplot_kw=dict(projection=Orthographic(95, 30))
    ...     )
    ...     ax.stock_img()
    ...     ax.coastlines()
    ...
    ...     airways["L888"].plot(
    ...         ax, linewidth=2, linestyle="solid", color="crimson"
    ...     )
    ...
    ...     for i, name in enumerate(airways["L888"].navaids):
    ...         navaids[name].plot(ax, s=20, marker="^", color="crimson")

    .. image:: _static/airways_l888.png
        :scale: 50%
        :alt: L888 route
        :align: center

    """

    cache_dir: Path

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self._data = data

    def download_data(self) -> None:  # coverage: ignore
        self._data = pd.read_csv(
            base_url + "/earth_awy.dat", sep=" ", header=-1
        )
        self._data.columns = ["route", "id", "navaid", "lat", "lon"]
        self._data.to_pickle(self.cache_dir / "traffic_airways.pkl")

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        if not (self.cache_dir / "traffic_airways.pkl").exists():
            self.download_data()
        else:
            logging.info("Loading airways database")
            self._data = pd.read_pickle(self.cache_dir / "traffic_airways.pkl")

        return self._data

    def __getitem__(self, name) -> Optional[Route]:
        output = self.data.query("route == @name").sort_values("id")
        if output.shape[0] == 0:
            return None
        ls = LineString(
            list((x["lon"], x["lat"]) for _, x in output.iterrows())
        )
        return Route(ls, name, list(output.navaid))

    def through(self, name: str) -> List[Route]:
        """Selects all routes going through the given navigational beacon.

        >>> airways.through('NARAK')
        [N869 (ROMAK, NASEP, NARAK),
         UN859 (PUMAL, ROCAN, LOMRA, GAI, NARAK, EVPOK, BALAN, AMB),
         UN869 (ELSAP, XOMBO, TIVLI, AGN, NARAK, NASEP, ROMAK, MINSO, MOKDI),
         UT122 (SECHE, NARAK, DITEV),
         UY155 (ETENU, NARAK),
         UZ365 (DIRMO, GUERE, NARAK)]

        """
        output: List[Route] = list()
        for id_ in self.search(name).data["route"]:
            item = self[id_]
            if item is not None:
                output.append(item)
        return output

    def search(self, name: str) -> "Airways":
        """
        Selects the subset of airways matching name in the route name or in the
        passed navigational beacon.

        >>> airways.extent('Switzerland').search("Z50")
                route   id   navaid  lat          lon
        101157  Z50     7    GERSA   47.039375    8.532114
        101158  Z50     8    KELIP   46.956194    8.761667
        101159  Z50     9    SOPER   46.889444    8.944444
        101160  Z50     10   PELAD   46.598889    9.725833
        101161  Z50     11   RESIA   46.478333    10.043333

        """
        output = self.__class__(
            self.data.query("route == @name.upper() or navaid == @name.upper()")
        )
        return output

    def extent(
        self,
        extent: Union[
            str, ShapelyMixin, Nominatim, Tuple[float, float, float, float]
        ],
        buffer: float = 0.5,
    ) -> "Airways":
        """
        Selects the subset of airways inside the given extent.

        The parameter extent may be passed as:

            - a string to query OSM Nominatim service;
            - the result of an OSM Nominatim query;
            - any kind of shape (including airspaces);
            - extents (west, east, south, north)

        >>> airways.extent('Switzerland')

        """
        if isinstance(extent, str):
            extent = location(extent)
        if isinstance(extent, ShapelyMixin):
            extent = extent.extent
        if isinstance(extent, Nominatim):
            extent = extent.extent

        west, east, south, north = extent

        output = self.query(
            f"{south - buffer} <= lat <= {north + buffer} and "
            f"{west - buffer} <= lon <= {east + buffer}"
        )
        return output
