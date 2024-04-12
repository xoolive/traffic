from __future__ import annotations

import codecs
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import rs1090

import numpy as np
import pandas as pd

from ...core import Traffic
from ...core.mixins import DataFrameMixin
from ...data import airports
from ...data.basic.airports import Airport

R = TypeVar("R", bound="RawData")
T = TypeVar("T")

_log = logging.getLogger(__name__)

MSG_SIZES = {0x31: 11, 0x32: 16, 0x33: 23, 0x34: 23}


def next_beast_msg(chunk_it: Iterable[bytes]) -> Iterator[bytes]:
    """Iterate in Beast binary feed.

    <esc> "1" : 6 byte MLAT timestamp, 1 byte signal level,
        2 byte Mode-AC
    <esc> "2" : 6 byte MLAT timestamp, 1 byte signal level,
        7 byte Mode-S short frame
    <esc> "3" : 6 byte MLAT timestamp, 1 byte signal level,
        14 byte Mode-S long frame
    <esc> "4" : 6 byte MLAT timestamp, status data, DIP switch
        configuration settings (not on Mode-S Beast classic)
    <esc><esc>: true 0x1a
    <esc> is 0x1a, and "1", "2" and "3" are 0x31, 0x32 and 0x33

    timestamp:
    wiki.modesbeast.com/Radarcape:Firmware_Versions#The_GPS_timestamp
    """
    data = b""
    for chunk in chunk_it:
        data += chunk
        while len(data) >= 23:
            it = data.find(0x1A)
            if it < 0:
                break
            data = data[it:]
            if len(data) < 23:
                break

            if data[1] in [0x31, 0x32, 0x33, 0x34]:
                # The tricky part here is to collapse all 0x1a 0x1a into single
                # 0x1a when they are part of a message (i.e. not followed by
                # "1", "2", "3" or "4")
                msg_size = MSG_SIZES[data[1]]
                ref_idx = 1
                idx = data[ref_idx:msg_size].find(0x1A)
                while idx != -1 and len(data) > msg_size:
                    start = ref_idx + idx
                    ref_idx = start + 1
                    if data[ref_idx] == 0x1A:
                        data = data[:start] + data[ref_idx:]
                    idx = data[ref_idx:msg_size].find(0x1A)
                if idx != -1 or len(data) < msg_size:
                    # calling for next buffer
                    break
                yield data[:msg_size]
                data = data[msg_size:]
            else:
                data = data[1:]
                _log.warning("Probably corrupted message")


def decode_time_default(
    msg: str, time_0: Optional[datetime] = None
) -> datetime:
    return datetime.now(timezone.utc)


def decode_time_radarcape(
    msg: str, time_0: Optional[datetime] = None
) -> datetime:
    now = datetime.now(timezone.utc)
    if time_0 is not None:
        now = time_0
    timestamp = int(msg[4:16], 16)

    nanos = timestamp & 0x00003FFFFFFF
    secs = timestamp >> 30
    ts = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ts += timedelta(seconds=secs, microseconds=nanos / 1000)
    if ts - timedelta(minutes=5) > now:
        ts -= timedelta(days=1)
    return ts


def decode_time_dump1090(
    msg: str, time_0: Optional[datetime] = None
) -> datetime:
    now = datetime.now(timezone.utc)
    if time_0 is not None:
        now = time_0
    else:
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)

    timestamp = int(msg[4:16], 16)
    # dump1090/net_io.c => time (in 12Mhz ticks)
    now += timedelta(seconds=timestamp / 12e6)

    return now


decode_time: dict[str, Callable[[str, Optional[datetime]], datetime]] = {
    "radarcape": decode_time_radarcape,
    "dump1090": decode_time_default,
    "default": decode_time_default,
}


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the to_be_stopped() condition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # self.daemon = True  # is it redundant?
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def to_be_stopped(self) -> bool:
        return self._stop_event.is_set()


class DumpFormat:
    def __init__(self, template: str, sep: str = ",") -> None:
        self.template = template
        self.sep = sep
        self.cols = list(x.strip() for x in template.split(sep))
        time_gen = (i for i, elt in enumerate(self.cols) if elt == "time")
        time_index = next(time_gen, None)
        if time_index is not None:
            self.time_index = time_index
        else:
            msg = "Format invalid: must contain 'time'"
            raise ValueError(msg)

        long_gen = (i for i, elt in enumerate(self.cols) if elt == "longmsg")
        self.msg_index = next(long_gen, None)
        self.splitmsg = slice(18, None)

        if self.msg_index is not None:
            return

        short_gen = (i for i, elt in enumerate(self.cols) if elt == "shortmsg")
        self.msg_index = next(short_gen, None)
        if self.msg_index is None:
            msg = "Format invalid: must contain either 'longmsg' or 'shortmsg'"
            raise ValueError(msg)

        self.splitmsg = slice(None)

    def get_timestamp(self, line: str) -> datetime:
        elts = line.split(self.sep)
        return datetime.fromtimestamp(
            float(elts[self.time_index].strip()),
            timezone.utc,
        )

    def get_msg(self, line: str) -> str:
        elts = line.split(self.sep)
        return elts[self.msg_index][self.splitmsg].strip()  # type: ignore


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
    def __init__(self, data: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
        super().__init__(data, *args, **kwargs)

        for column_name in ["timestamp", "mintime", "maxtime"]:
            if column_name in self.data.columns:
                if data[column_name].dtype == np.float64:
                    self.data = self.data.assign(
                        **{
                            column_name: pd.to_datetime(
                                data[column_name], unit="s", utc=True
                            )
                        }
                    )

    def __add__(self: R, other: R) -> R:
        return self.__class__.from_list([self, other])

    @classmethod
    def from_list(cls: Type[R], elts: Iterable[Optional[R]]) -> R:
        res = cls(
            pd.concat(list(x.data for x in elts if x is not None), sort=False)
        )
        return res.sort_values("mintime")

    def decode(
        self,
        reference: Union[None, str, Airport, Tuple[float, float]] = None,
    ) -> Optional[Traffic]:
        if isinstance(reference, str):
            reference = airports[reference]
        if hasattr(reference, "latlon"):
            reference = reference.latlon  # type: ignore

        decoded = rs1090.decode(
            self.data.rawmsg,
            self.data.timestamp.astype("int64") * 1e-9,
            reference=reference,
        )

        if len(decoded) == 0:
            return None

        df = pd.concat(
            # 5000 is a good batch size for fast loading!
            pd.DataFrame.from_records(d)
            for d in rs1090.batched(decoded, 5000)
        )
        df = df.assign(
            timestamp=pd.to_datetime(df.timestamp, unit="s", utc=True)
        )
        return Traffic(df)

    def assign_beast(self, time_fmt: str = "dump1090") -> "RawData":
        # Only one time encoder implemented for now
        encoder = encode_time.get(time_fmt, encode_time_dump1090)

        return self.assign(encoded_time=lambda df: encoder(df.mintime)).assign(
            beast=lambda df: "@" + df.encoded_time + df.rawmsg
        )

    def to_beast(self, time_fmt: str = "dump1090") -> pd.Series:
        return self.assign_beast(time_fmt).data["beast"]

    @classmethod
    def from_dump1090_output(
        cls,
        filename: str | Path,
        reference: Union[str, Airport, tuple[float, float]],
        reference_time: str = "now",
    ) -> Optional[Traffic]:  # coverage: ignore
        """Decode raw messages dumped from `dump1090
        <https://github.com/MalcolmRobb/dump1090/>`_ with option mlat

        :param filename: the path to the file containing the data
        :param reference: the reference location, as specified above

        .. warning::

            dump1090 must be run the ``--mlat`` option.

        """
        now = pd.Timestamp(reference_time, tz="utc")

        if isinstance(reference, str):
            reference = airports[reference]
        if hasattr(reference, "latlon"):
            reference = reference.latlon  # type: ignore

        filename = Path(filename)
        b_content = filename.read_bytes()
        try:
            all_lines = b_content.decode()
        except UnicodeDecodeError:
            all_lines = codecs.encode(b_content, "hex").decode()

        if all_lines.startswith("@"):  # dump1090 with --mlat option
            msgs: list[tuple[pd.Timestamp, str]] = list(
                (
                    now + timedelta(seconds=int(line[1:13], 16) / 12e6),
                    line[13:-1],
                )
                for line in all_lines.split("\n")
                if line.startswith("@")
            )
            timestamps, msg_iter = zip(*msgs)

            decoded = rs1090.decode(
                msg_iter,
                [t.timestamp() for t in timestamps],
                reference=reference,
            )

        elif all_lines.startswith("1a"):  # xxd -p on radarcape dump
            content = bytes.fromhex(all_lines.replace("\n", ""))
            raw_msgs = list(
                "".join(["{:02x}".format(t) for t in bin_msg])
                for bin_msg in next_beast_msg([content])
                if len(bin_msg) == 23
            )
            rc_timestamps = [
                decode_time_radarcape(msg, now).timestamp() for msg in raw_msgs
            ]
            msg_list = [msg[18:] for msg in raw_msgs]

            decoded = rs1090.decode(
                msg_list,
                rc_timestamps,
                reference=reference,
            )
        else:
            return None

        if len(decoded) == 0:
            return None

        df = pd.concat(
            # 5000 is a good batch size for fast loading!
            pd.DataFrame.from_records(d)
            for d in rs1090.batched(decoded, 5000)
        )
        df = df.assign(
            timestamp=pd.to_datetime(df.timestamp, unit="s", utc=True)
        )
        return Traffic(df)

    @classmethod
    def process_file(
        cls: Type[R],
        filename: str | Path,
        reference: str | Airport | tuple[float, float],
        template: str = "time, longmsg",
        sep: str = ",",
    ) -> Optional[Traffic]:
        """Decode raw messages dumped in a text file.

        The file should contain for each line at least a timestamp and an
        hexadecimal message, as a CSV-like format.

        :param filename: the path to the file containing the data

        :param reference: the reference location, as specified above

        :param template: the header explaining how data is organised

            Three parameters are accepted:

            - ``time`` represents the timestamp in seconds (float)
            - ``shortmsg`` represents the regular version of the ADS-B
              hexadecimal message (messages of length 28 for ADS-B)
            - ``longmsg`` represents messages containing timestamp information
              as a prefix, as dumped by many decoding softwares, such as
              `dump1090 <https://github.com/MalcolmRobb/dump1090/>`_ or other
              receivers.

            By default, the expected format is ``time, longmsg``
        """

        if isinstance(filename, str):
            filename = Path(filename)

        dumpformat = DumpFormat(template, sep)

        if isinstance(reference, str):
            reference = airports[reference]
        if hasattr(reference, "latlon"):
            reference = reference.latlon  # type: ignore

        with filename.open("r") as fh:
            all_lines = fh.readlines()
            msgs = list(
                (
                    dumpformat.get_timestamp(line),
                    dumpformat.get_msg(line),
                )
                for line in all_lines
            )

            timestamps, msg_iter = zip(*msgs)

        decoded = rs1090.decode(
            msg_iter,
            [t.timestamp() for t in timestamps],
            reference=reference,
        )

        if len(decoded) == 0:
            None

        df = pd.concat(
            # 5000 is a good batch size for fast loading!
            pd.DataFrame.from_records(d)
            for d in rs1090.batched(decoded, 5000)
        )
        df = df.assign(
            timestamp=pd.to_datetime(df.timestamp, unit="s", utc=True)
        )
        return Traffic(df)
