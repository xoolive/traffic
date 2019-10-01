import warnings
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import pandas as pd

from ...basic.navaid import Navaids

# https://github.com/python/mypy/issues/2511
T = TypeVar("T", bound="NMNavaids")


class NMNavaids(Navaids):
    @classmethod
    def from_file(
        cls: Type[T], filename: Union[Path, str], **kwargs
    ) -> Optional[T]:

        filename = Path(filename)

        nnpt_file = next(filename.glob("AIRAC_*.nnpt"), None)
        if nnpt_file is None:
            msg = f"No AIRAC_*.nnpt file found in {filename}"
            if kwargs.get("error", True):
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)
            return None

        return cls(
            pd.read_csv(
                nnpt_file,
                sep=";",
                names=["name", "type", "lat", "lon", "description"],
                skiprows=1,
            )
        )
