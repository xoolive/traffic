from typing import Iterable, Optional, Type, TypeVar

from pitot.geodesy import distance
from pyModeS.decoder.flarm import DecodedMessage, flarm
from tqdm.rich import tqdm
from typing_extensions import Annotated

import pandas as pd

from ...core.mixins import DataFrameMixin
from ...core.traffic import Traffic

T = TypeVar("T", bound="FlarmData")


def receiver_position(
    msg: DecodedMessage,
    threshold: Annotated[float, "m"] = 100_000,
) -> bool:
    delta_d: Annotated[float, "m"] = distance(  # in meters
        msg["latitude"],
        msg["longitude"],
        msg["sensorLatitude"],
        msg["sensorLongitude"],
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
        msgs = (
            field
            for _, line in tqdm(self.data.iterrows(), total=self.data.shape[0])
            if (
                field := flarm(
                    line.timeatplane,
                    line.rawmessage,
                    line.sensorlatitude,
                    line.sensorlongitude,
                    sensorName=line.sensorname,
                )
            )
            is not None
        )

        decoded = pd.DataFrame.from_records(msgs).eval(
            "timestamp = @pd.to_datetime(timestamp, unit='s', utc=True)"
        )

        if decoded.shape[0] == 0:
            return None

        mask = receiver_position(decoded)
        return Traffic(decoded.loc[mask])
