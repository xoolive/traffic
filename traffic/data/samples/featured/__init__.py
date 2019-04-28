from pathlib import Path

from .. import _assign, get_flight

_current_dir = Path(__file__).parent
__all__ = list(f.stem[:-5] for f in _current_dir.glob("*.json.gz"))


def __getattr__(name: str):
    if name not in __all__:
        return None
    return get_flight(name, _current_dir)


traffic = sum(
    get_flight(name, _current_dir).pipe(lambda x: _assign(x, name))
    for name in __all__
)
