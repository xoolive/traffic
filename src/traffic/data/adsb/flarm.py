from typing import Iterable, Optional, Type, TypeVar

import rs1090
from pitot.geodesy import distance
from typing_extensions import Annotated

import pandas as pd

from ...core.mixins import DataFrameMixin
from ...core.traffic import Traffic

T = TypeVar("T", bound="FlarmData")


def receiver_position(
    msg: pd.DataFrame,
    threshold: Annotated[float, "m"] = 100_000,
) -> bool:
    delta_d: Annotated[float, "m"] = distance(  # in meters
        msg["latitude"],
        msg["longitude"],
        msg["reference_lat"],
        msg["reference_lon"],
    )
    return delta_d < threshold


class FlarmData(DataFrameMixin):
    def __add__(self: T, other: T) -> T:
        return self.__class__.from_list([self, other])

    @classmethod
    def from_list(cls: Type[T], elts: Iterable[Optional[T]]) -> T:
        res = cls(
            pd.concat(list(x.data for x in elts if x is not None), sort=False)
        )
        return res.sort_values("mintime")

    def decode(self) -> Optional[Traffic]:
        decoded = rs1090.flarm(
            self.data.rawmessage.values,
            self.data.timeatserver.values,
            self.data.sensorlatitude,
            self.data.sensorlongitude,
        )
        if len(decoded) == 0:
            return None

        # 5000 is a good batch size for fast loading!
        df = pd.concat(
            pd.DataFrame.from_records(d) for d in rs1090.batched(decoded, 5000)
        )
        df = df.assign(
            timestamp=pd.to_datetime(df.timestamp, unit="s", utc=True)
        )

        mask = receiver_position(df)
        return Traffic(df.loc[mask])
