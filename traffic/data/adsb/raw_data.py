# fmt: off

from typing import (
    Any, Callable, Dict, Iterable, Optional, Tuple, Type, TypeVar, Union, cast
)

import numpy as np
import pandas as pd

from pyModeS import adsb
from tqdm.autonotebook import tqdm

from ...core import Traffic
from ...core.mixins import DataFrameMixin
from ...core.structure import Airport
from ...data import ModeS_Decoder

# fmt: on

T = TypeVar("T", bound="RawData")


def encode_time_dump1090(times: pd.Series) -> pd.Series:
    if isinstance(times.iloc[0], pd.datetime):
        times = times.astype(np.int64) * 1e-9
    ref_time = times.iloc[0]
    rel_times = times - ref_time
    rel_times = rel_times * 12e6
    rel_times = rel_times.apply(lambda row: hex(int(row))[2:].zfill(12))
    return rel_times


encode_time: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "dump1090": encode_time_dump1090
}


class RawData(DataFrameMixin):
    def __add__(self: T, other: T) -> T:
        return self.__class__.from_list([self, other])

    @classmethod
    def from_list(cls: Type[T], elts: Iterable[Optional[T]]) -> T:
        res = cls(
            pd.concat(list(x.data for x in elts if x is not None), sort=False)
        )
        return res.sort_values("mintime")

    def decode(
        self: T,
        reference: Union[None, str, Airport, Tuple[float, float]] = None,
        *,
        uncertainty: bool = False,
        progressbar: Union[bool, Callable[[Iterable], Iterable]] = True,
        progressbar_kw: Optional[Dict[str, Any]] = None,
        redefine_mag: int = 10,
    ) -> Optional[Traffic]:

        decoder = ModeS_Decoder(reference)
        redefine_freq = 2 ** redefine_mag - 1

        if progressbar is True:
            if progressbar_kw is None:
                progressbar_kw = dict()
            progressbar = lambda x: tqdm(  # noqa: E731
                x, total=self.data.shape[0], **progressbar_kw
            )
        elif progressbar is False:
            progressbar = lambda x: x  # noqa: E731

        progressbar = cast(Callable[[Iterable], Iterable], progressbar)

        data = self.data.rename(  # fill with other common renaming rules
            columns={
                "mintime": "timestamp",
                "time": "timestamp",
                "groundspeed": "spd",
                "speed": "spd",
                "altitude": "alt",
                "track": "trk",
            }
        )

        use_extra = all(x in data.columns for x in ["alt", "spd", "trk"])

        for i, (_, line) in progressbar(enumerate(data.iterrows())):

            extra = (
                dict(
                    spd=line.spd,
                    trk=line.trk,
                    alt=line.alt,
                    uncertainty=uncertainty,
                )
                if use_extra
                else dict(uncertainty=uncertainty)
            )

            decoder.process(line.timestamp, line.rawmsg, **extra)

            if i & redefine_freq == redefine_freq:
                decoder.redefine_reference(line.timestamp)

        return decoder.traffic

    def assign_type(self) -> "RawData":
        def get_typecode(msg: Union[bytes, str]) -> Optional[int]:
            tc = adsb.typecode(msg)
            if 9 <= tc <= 18:
                return 3
            elif tc == 19:
                return 4
            elif 1 <= tc <= 4:
                return 1
            else:
                return None

        return self.assign(msg_type=lambda df: df.rawmsg.apply(get_typecode))

    def assign_beast(self, time_fmt: str = "dump1090") -> "RawData":

        # Only one time encoder implemented for now
        encoder = encode_time.get(time_fmt, encode_time_dump1090)

        return self.assign(encoded_time=lambda df: encoder(df.mintime)).assign(
            beast=lambda df: "@" + df.encoded_time + df.rawmsg
        )

    def to_beast(self, time_fmt: str = "dump1090") -> pd.Series:

        return self.assign_beast(time_fmt).data["beast"]
