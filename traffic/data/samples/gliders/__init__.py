from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

from .. import get_flight

if TYPE_CHECKING:
    from ....core import Flight, Traffic

_current_dir = Path(__file__).parent
__all__ = sorted(f.stem[:-5] for f in _current_dir.glob("*.json.gz"))


def __getattr__(name: str) -> Union[None, "Flight", "Traffic"]:
    if name == "traffic":
        return Traffic.from_flights(
            cast(Flight, get_flight(name, _current_dir)).assign_id(name)
            for name in __all__
        )

    if name not in __all__:
        return None
    return get_flight(name, _current_dir)
