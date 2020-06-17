import pandas as pd

from ..basic.navaid import Navaids
from . import ADDS_FAA_OpenData


class FAA_Designated_Points(Navaids):
    name: str = "faa_designated_points"

    @property
    def available(self) -> bool:
        return True


class Designated_Points(ADDS_FAA_OpenData):

    id_ = "861043a88ff4486c97c3789e7dcdccc6_0"
    filename = "faa_designated_points.json"

    def get_data(self) -> FAA_Designated_Points:
        return FAA_Designated_Points(
            pd.DataFrame.from_records(
                {
                    **x["properties"],
                    **{
                        "longitude": x["geometry"]["coordinates"][0],
                        "latitude": x["geometry"]["coordinates"][1],
                    },
                }
                for x in self.json_contents()["features"]
                if x["geometry"] is not None
            )
            .drop(
                columns=[
                    "OBJECTID",
                    "REMARKS",
                    "NOTES_ID",
                    "MIL_CODE",
                    "REPATC",
                    "MAGVAR_DT",
                    "ONSHORE",
                    "STRUCTURE",
                    "REFFAC",
                    "MRA_VAL",
                    "MRA_UOM",
                    "STATE",
                    "COUNTRY",
                    "AK_LOW",
                    "AK_HIGH",
                    "US_LOW",
                    "US_HIGH",
                    "US_AREA",
                    "LATITUDE",
                    "LONGITUDE",
                    "PACIFIC",
                ]
            )
            .rename(columns=str.lower)
            .assign(description=None)
            .rename(
                columns=dict(
                    ident="name", magvar="magnetic_variation", type_code="type"
                )
            )[
                [
                    "name",
                    "type",
                    "latitude",
                    "longitude",
                    "magnetic_variation",
                    "description",
                    "global_id",
                ]
            ]
        )
