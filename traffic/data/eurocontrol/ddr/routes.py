# fmt: off

from pathlib import Path
from typing import Optional

import pandas as pd

from ...basic.airways import Airways

# fmt: on


class NMRoutes(Airways):
    nm_path: Optional[Path] = None

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
