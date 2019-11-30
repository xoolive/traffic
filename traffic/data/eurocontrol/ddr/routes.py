# fmt: off

from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ...basic.airways import Airways

# fmt: on


class NMRoutes(Airways):
    nm_path: Optional[Path] = None
    name: str = "nm_airways"

    @lru_cache()
    def _ipython_key_completions_(self) -> Set[str]:
        return set(self.data.route)

    @property
    def available(self) -> bool:
        return self.nm_path is not None

    @property
    def data(self):
        if self._data is None:
            msg = f"Edit config file with NM directory"

            if self.nm_path is None:
                raise RuntimeError(msg)

            route_file = next(self.nm_path.glob("AIRAC_*.routes"), None)
            if route_file is None:
                raise RuntimeError(
                    f"No AIRAC*.routes file found in {self.nm_path}"
                )

            from ....data import nm_navaids

            assert nm_navaids is not None
            self._data = pd.read_csv(
                route_file,
                sep=";",
                skiprows=1,
                usecols=["route", "type", "navaid", "id"],
                names=["_", "route", "type", "_1", "_2", "navaid", "_3", "id"],
            ).merge(nm_navaids.data, left_on=["navaid"], right_on=["name"])

        return self._data
