from typing import Any, Dict

from . import ADDS_FAA_OpenData


class Ats_Route(ADDS_FAA_OpenData):
    id_ = "acf64966af5f48a1a40fdbcb31238ba7_0"
    filename = "faa_ats_route.json"

    def get_data(self) -> Dict[str, Any]:
        return self.json_contents()
