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


# this is here for compatibility with the <=2.12.0 API to allow for stuff like
# `from traffic.data.faa import class_airspace`
def __getattr__(name: str) -> Any:
    for cls in __all__:
        if cls.lower() == name.lower():
            return globals()[cls]()

    raise AttributeError()
