import importlib
from typing import Any

from ._airspaces import (
    Airspace_Boundary,
    Class_Airspace,
    Prohibited_Airspace,
    Route_Airspace,
    Special_Use_Airspace,
)
from ._ats_route import Ats_Route
from ._designated_points import Designated_Points
from ._navaid_components import Navaid_Components

__all__ = [
    "Airspace_Boundary",
    "Ats_Route",
    "Class_Airspace",
    "Designated_Points",
    "Navaid_Components",
    "Prohibited_Airspace",
    "Route_Airspace",
    "Special_Use_Airspace",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = importlib.import_module("._" + name, package="traffic.data.faa")
        return getattr(mod, name.title())()

    raise AttributeError()
