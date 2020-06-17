import pandas as pd

from ..basic.navaid import Navaids
from . import ADDS_FAA_OpenData


class Navaid_Components(ADDS_FAA_OpenData):

    id_ = "c9254c171b6741d3a5e494860761443a_0"
    filename = "faa_navaid_components.json"

    def get_data(self) -> Navaids:
        return Navaids(
            pd.DataFrame.from_records(
                {
                    **x["properties"],
                    **{
                        "longitude": x["geometry"]["coordinates"][0],
                        "latitude": x["geometry"]["coordinates"][1],
                    },
                }
                for x in self.json_contents()["features"]
            )
            .drop(
                columns=[
                    "WKHR_CODE",
                    "WKHR_RMK",
                    "STATUS",
                    "VOICE",
                    "SLAVEVAR",
                    "PRIVATE",
                    "AWYSTRUC",
                    "CHANNEL",
                    "AK_LOW",
                    "AK_HIGH",
                    "US_LOW",
                    "US_HIGH",
                    "US_AREA",
                    "MAGVAR_DAT",
                    "LATITUDE",
                    "LONGITUDE",
                    "TYPE_CODE",
                    "PACIFIC",
                    "OBJECTID",
                ]
            )
            .rename(columns=str.lower)
            .merge(
                pd.DataFrame.from_records(
                    [
                        {"nav_type": 1, "type": "NDB"},
                        {"nav_type": 2, "type": "DME"},
                        {"nav_type": 3, "type": "VOR"},
                        {"nav_type": 4, "type": "TACAN"},
                    ]
                )
            )
            .rename(
                columns=dict(
                    elevation="altitude",
                    ident="name",
                    name="description",
                    magvar="magnetic_variation",
                )
            )[
                [
                    "name",
                    "type",
                    "latitude",
                    "longitude",
                    "altitude",
                    "frequency",
                    "magnetic_variation",
                    "description",
                    "global_id",
                    # "navsys_id",
                ]
            ]
        )
