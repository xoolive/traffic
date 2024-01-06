from typing import Iterable, TypeVar, Any

from traffic import tqdm_style
from .types import ProgressbarType


# Choose the appropriate tqdm based on the style
if tqdm_style == "notebook":
    from tqdm.notebook import tqdm
elif tqdm_style == "rich":
    from tqdm.rich import tqdm
elif tqdm_style == "auto":
    from tqdm.auto import tqdm
elif tqdm_style == "silence":

    def tqdm(iterable: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:
        return iterable  # Dummy tqdm function
else:
    from tqdm import tqdm  # noqa: F401
