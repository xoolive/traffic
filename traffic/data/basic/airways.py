import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from ...core.mixins import GeoDBMixin
from ...core.structure import Route

__github_url = "https://raw.githubusercontent.com/"
base_url = __github_url + "xoolive/traffic/master/data/navdata"

BoundsType = Union[BaseGeometry, Tuple[float, float, float, float]]


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
    alternatives: Dict[str, "Airways"] = dict()
    name: str = "default"

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        self._data = data

    def __new__(cls, data: Optional[pd.DataFrame] = None) -> "Airways":
        instance = super().__new__(cls)
        if instance.available:
            Airways.alternatives[cls.name] = instance
        return instance

    def download_data(self) -> None:  # coverage: ignore
        from .. import session

        c = session.get(base_url + "/earth_awy.dat")
        c.raise_for_status()
        b = BytesIO(c.content)
        self._data = pd.read_csv(b, sep=" ", header=None)
        self._data.columns = ["route", "id", "navaid", "latitude", "longitude"]
        self._data.to_pickle(self.cache_dir / "traffic_airways.pkl")

    @property
    def available(self) -> bool:
        return True

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data

        if not (self.cache_dir / "traffic_airways.pkl").exists():
            self.download_data()
        else:
            logging.info("Loading airways database")
            self._data = pd.read_pickle(self.cache_dir / "traffic_airways.pkl")

        if self._data is not None:
            self._data = self._data.rename(
                columns=dict(lat="latitude", lon="longitude")
            )

        return self._data

    def __getitem__(self, name) -> Optional[Route]:
        output = self.data.query("route == @name").sort_values("id")
        if output.shape[0] == 0:
            return None
        ls = LineString(
            list((x["longitude"], x["latitude"]) for _, x in output.iterrows())
        )
        return Route(ls, name, list(output.navaid))

    def global_get(self, name) -> Optional[Route]:
        """Search for a route from all alternative data sources."""
        for _key, value in self.alternatives.items():
            alt = value[name]
            if alt is not None:
                return alt
        return None

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
