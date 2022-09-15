# mypy: ignore-errors
# This line can be removed when typing in pyModeS is complete and released.

from __future__ import annotations

import heapq
import logging
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    TextIO,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import pyModeS as pms
from tqdm.rich import tqdm

import pandas as pd

from ...core import Flight, Traffic
from ...data.basic.airports import Airport

Decoder = TypeVar("Decoder", bound="ModeS_Decoder")

_log = logging.getLogger(__name__)

MSG_SIZES = {0x31: 11, 0x32: 16, 0x33: 23, 0x34: 23}


def next_beast_msg(chunk_it: Iterator[bytes]) -> Iterator[bytes]:
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
    "dump1090": decode_time_dump1090,
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


# total=False means that some keys can be absent
class Entry(TypedDict, total=False):
    timestamp: datetime
    icao24: str
    callsign: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[int]
    geoaltitude: Optional[int]
    groundspeed: Optional[int]
    track: Optional[float]
    vertical_rate: Optional[int]
    onground: Optional[bool]
    squawk: Optional[int]
    # BDS 4,0
    selected_fms: Optional[int]
    selected_mcp: Optional[int]
    barometric_setting: Optional[int]
    # BDS 4,4
    humidity: Optional[int]
    pressure: Optional[int]
    temperature: Optional[int]
    turbulence: Optional[int]
    windspeed: Optional[int]
    winddirection: Optional[int]
    # BDS 4,5
    wind_shear: Optional[int]
    microburst: Optional[int]
    icing: Optional[int]
    wake_vortex: Optional[int]
    radio_height: Optional[int]
    # BDS 5,0
    roll: Optional[float]
    TAS: Optional[int]
    track_rate: Optional[float]
    # BDS 6,0
    IAS: Optional[int]
    heading: Optional[float]
    Mach: Optional[float]
    vertical_rate_barometric: Optional[int]
    vertical_rate_inertial: Optional[int]
    # Uncertainty
    HPL: Optional[int]
    RCu: Optional[int]
    RCv: Optional[int]
    HCR: Optional[int]
    VPL: Optional[int]
    HVE: Optional[int]
    VVE: Optional[int]
    HFM: Optional[int]
    VFM: Optional[int]
    EPU: Optional[int]
    VEPU: Optional[int]
    NACp: Optional[int]
    version: Optional[int]
    pHCR: Optional[int]
    pVPL: Optional[int]
    sil_base: Optional[int]


class Aircraft(object):
    def __init__(self, icao24: str, lat0: float, lon0: float) -> None:
        self.icao24 = icao24
        self._callsign: Optional[str] = None
        self._flight: Optional[Flight] = None
        self.cumul: list[Entry] = []

        self.t0: Optional[datetime] = None
        self.t1: Optional[datetime] = None
        self.tpos: Optional[datetime] = None

        self.m0: Optional[str] = None
        self.m1: Optional[str] = None

        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.alt: Optional[int] = None
        self.trk: Optional[float] = None
        self.spd: Optional[int] = None

        self.lat0: float = lat0
        self.lon0: float = lon0

        self.version: Optional[int] = None
        self.nic_a: Optional[int] = None
        self.nic_bc: Optional[int] = None
        self.nic_s: Optional[int] = None

        self.lock = threading.Lock()

    @property
    def flight(self) -> Optional[Flight]:
        with self.lock:  # access then clear not thread-safe, hence the lock
            df = pd.DataFrame.from_records(self.cumul)
            self.cumul.clear()

        if self._flight is not None:
            if len(df) > 0:
                df = pd.concat([self._flight.data, df], sort=False)
            else:
                df = self._flight.data
            if self.version is not None:
                # remove columns added by nuc_p, nuc_r
                if "HPL" in df.columns:
                    df = df.drop(columns=["HPL", "RCu", "RCv"])
                if "HVE" in df.columns:
                    df = df.drop(columns=["HVE", "VVE"])

        if len(df) == 0:
            return None

        if "callsign" in set(df.columns):
            self._flight = Flight(
                df.assign(
                    callsign=df.callsign.replace("", None)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                )
            )
        else:
            self._flight = Flight(df)

        return self._flight

    @property
    def callsign(self) -> Optional[str]:
        return self._callsign

    @callsign.setter
    def callsign(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        callsign = pms.adsb.callsign(msg).strip("_")
        if callsign == "":
            return
        self._callsign = callsign
        with self.lock:
            self.cumul.append(
                dict(timestamp=t, icao24=self.icao24, callsign=self._callsign)
            )

    @property
    def speed(self) -> None:
        pass

    @speed.setter
    def speed(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        vdata = pms.adsb.velocity(msg)
        if vdata is None:
            return

        spd, trk, roc, tag = vdata
        if tag != "GS":
            # does it ever happen...
            return
        if (spd is None) or (trk is None):
            return

        self.spd = spd
        self.trk = trk

        delta = pms.adsb.altitude_diff(msg)

        with self.lock:
            self.cumul.append(
                dict(
                    timestamp=t,
                    icao24=self.icao24,
                    groundspeed=spd,
                    track=trk,
                    vertical_rate=roc,
                )
            )
            if delta is not None and self.alt is not None:
                self.cumul[-1]["geoaltitude"] = self.alt + delta

    @property
    def position(self) -> None:
        pass

    @position.setter
    def position(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        oe = pms.adsb.oe_flag(msg)
        setattr(self, "m" + str(oe), msg)
        setattr(self, "t" + str(oe), t)

        if (
            self.t0 is not None
            and self.t1 is not None
            and abs((self.t0 - self.t1).total_seconds()) < 10
        ):
            latlon = pms.adsb.position(
                self.m0, self.m1, self.t0, self.t1, self.lat0, self.lon0
            )
        else:
            latlon = None

        if latlon is not None:
            self.tpos = t
            self.lat, self.lon = latlon
            self.alt = pms.adsb.altitude(msg)

            with self.lock:
                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        latitude=self.lat,
                        longitude=self.lon,
                        altitude=self.alt,
                        onground=False,
                    )
                )

    @property
    def surface(self) -> None:
        pass

    @surface.setter
    def surface(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        self.lat, self.lon = pms.adsb.surface_position_with_ref(
            msg, self.lat0, self.lon0
        )
        speed, track, _, speed_type, *_ = pms.adsb.surface_velocity(msg)
        if speed_type != "GS":
            _log.warn(f"Ground airspeed for aircraft {self.icao24}")

        # This helps updating current representations
        self.spd = speed
        self.trk = track

        with self.lock:
            self.cumul.append(
                dict(
                    timestamp=t,
                    icao24=self.icao24,
                    latitude=self.lat,
                    longitude=self.lon,
                    groundspeed=speed,
                    track=track,
                    onground=True,
                )
            )

    @property
    def altcode(self) -> None:
        pass

    @altcode.setter
    def altcode(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        from pyModeS import hex2bin

        if set(hex2bin(msg)[19:32]) in [{"0"}, {"1"}]:
            return
        self.alt = pms.common.altcode(msg)
        with self.lock:
            self.cumul.append(
                dict(timestamp=t, icao24=self.icao24, altitude=self.alt)
            )

    @property
    def idcode(self) -> None:
        pass

    @idcode.setter
    def idcode(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        from pyModeS import hex2bin

        if set(hex2bin(msg)[19:32]) in [{"0"}, {"1"}]:
            return
        idcode = pms.common.idcode(msg)
        with self.lock:
            self.cumul.append(
                dict(
                    timestamp=t,
                    icao24=self.icao24,
                    squawk=idcode,
                )
            )

    @property
    def bds20(self) -> None:
        pass

    @bds20.setter
    def bds20(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        callsign = pms.commb.cs20(msg).strip("_")
        if callsign == "":
            return
        self._callsign = callsign
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {  # type: ignore
                    **last_entry,
                    **dict(callsign=self._callsign),
                }
            else:
                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        callsign=self._callsign,
                    )
                )

    @property
    def bds40(self) -> None:
        pass

    @bds40.setter
    def bds40(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {  # type: ignore
                    **last_entry,
                    **dict(
                        # FMS selected altitude (ft)
                        selected_fms=pms.commb.selalt40fms(msg),
                        # MCP/FCU selected altitude (ft)
                        selected_mcp=pms.commb.selalt40mcp(msg),
                        # Barometric pressure (mb)
                        barometric_setting=pms.commb.p40baro(msg),
                    ),
                }

            else:
                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        # FMS selected altitude (ft)
                        selected_fms=pms.commb.selalt40fms(msg),
                        # MCP/FCU selected altitude (ft)
                        selected_mcp=pms.commb.selalt40mcp(msg),
                        # Barometric pressure (mb)
                        barometric_setting=pms.commb.p40baro(msg),
                    )
                )

    @property
    def bds44(self) -> None:
        pass

    @bds44.setter
    def bds44(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        wind = pms.commb.wind44(msg)
        wind = wind if wind is not None else (None, None)
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF 5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {  # type: ignore
                    **last_entry,
                    **dict(
                        # Humidity (%)
                        humidity=pms.commb.hum44(msg),
                        # Average static pressure (hPa)
                        pressure=pms.commb.p44(msg),
                        # Static air temperature (C)
                        temperature=pms.commb.temp44(msg),
                        turbulence=pms.commb.turb44(msg),
                        # Wind speed (kt) and direction (true) (deg)
                        windspeed=wind[0],
                        winddirection=wind[1],
                    ),
                }

            else:
                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        # Humidity (%)
                        humidity=pms.commb.hum44(msg),
                        # Average static pressure (hPa)
                        pressure=pms.commb.p44(msg),
                        # Static air temperature (C)
                        temperature=pms.commb.temp44(msg),
                        turbulence=pms.commb.turb44(msg),
                        # Wind speed (kt) and direction (true) (deg)
                        windspeed=wind[0],
                        winddirection=wind[1],
                    )
                )

    @property
    def bds45(self) -> None:
        pass

    @bds45.setter
    def bds45(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF 5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {  # type: ignore
                    **last_entry,
                    **dict(
                        # Turbulence level (0-3)
                        turbulence=pms.commb.turb45(msg),
                        # Wind shear level (0-3)
                        wind_shear=pms.commb.ws45(msg),
                        # Microburst level (0-3)
                        microburst=pms.commb.mb45(msg),
                        # Icing level (0-3)
                        icing=pms.commb.ic45(msg),
                        # Wake vortex level (0-3)
                        wake_vortex=pms.commb.wv45(msg),
                        # Static air temperature (C)
                        temperature=pms.commb.temp45(msg),
                        # Average static pressure (hPa)
                        pressure=pms.commb.p45(msg),
                        # Radio height (ft)
                        radio_height=pms.commb.rh45(msg),
                    ),
                }

            else:
                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        # Turbulence level (0-3)
                        turbulence=pms.commb.turb45(msg),
                        # Wind shear level (0-3)
                        wind_shear=pms.commb.ws45(msg),
                        # Microburst level (0-3)
                        microburst=pms.commb.mb45(msg),
                        # Icing level (0-3)
                        icing=pms.commb.ic45(msg),
                        # Wake vortex level (0-3)
                        wake_vortex=pms.commb.wv45(msg),
                        # Static air temperature (C)
                        temperature=pms.commb.temp45(msg),
                        # Average static pressure (hPa)
                        pressure=pms.commb.p45(msg),
                        # Radio height (ft)
                        radio_height=pms.commb.rh45(msg),
                    )
                )

    @property
    def bds50(self) -> None:
        pass

    @bds50.setter
    def bds50(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {  # type: ignore
                    **last_entry,
                    **dict(
                        # Ground speed (kt)
                        groundspeed=pms.commb.gs50(msg),
                        # Roll angle (deg)
                        roll=pms.commb.roll50(msg),
                        # True airspeed (kt)
                        TAS=pms.commb.tas50(msg),
                        # True track angle (deg)
                        track=pms.commb.trk50(msg),
                        # Track angle rate (deg/sec)
                        track_rate=pms.commb.rtrk50(msg),
                    ),
                }

            else:

                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        # Ground speed (kt)
                        groundspeed=pms.commb.gs50(msg),
                        # Roll angle (deg)
                        roll=pms.commb.roll50(msg),
                        # True airspeed (kt)
                        TAS=pms.commb.tas50(msg),
                        # True track angle (deg)
                        track=pms.commb.trk50(msg),
                        # Track angle rate (deg/sec)
                        track_rate=pms.commb.rtrk50(msg),
                    )
                )

    @property
    def bds60(self) -> None:
        pass

    @bds60.setter
    def bds60(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {  # type: ignore
                    **last_entry,
                    **dict(
                        # Indicated airspeed (kt)
                        IAS=pms.commb.ias60(msg),
                        # Magnetic heading (deg)
                        heading=pms.commb.hdg60(msg),
                        # Mach number (-)
                        Mach=pms.commb.mach60(msg),
                        # Barometric altitude rate (ft/min)
                        vertical_rate_barometric=pms.commb.vr60baro(msg),
                        # Inertial vertical speed (ft/min)
                        vertical_rate_inertial=pms.commb.vr60ins(msg),
                    ),
                }

            else:
                self.cumul.append(
                    dict(
                        timestamp=t,
                        icao24=self.icao24,
                        # Indicated airspeed (kt)
                        IAS=pms.commb.ias60(msg),
                        # Magnetic heading (deg)
                        heading=pms.commb.hdg60(msg),
                        # Mach number (-)
                        Mach=pms.commb.mach60(msg),
                        # Barometric altitude rate (ft/min)
                        vertical_rate_barometric=pms.commb.vr60baro(msg),
                        # Inertial vertical speed (ft/min)
                        vertical_rate_inertial=pms.commb.vr60ins(msg),
                    )
                )

    @property
    def nuc_p(self) -> None:
        pass

    @nuc_p.setter
    def nuc_p(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            hpl, rcu, rcv = pms.adsb.nuc_p(msg)
            current = dict(
                # Horizontal Protection Limit
                HPL=hpl,
                # 95% Containment Radius on horizontal position error
                RCu=rcu,
                # 95% Containment Radius on vertical position error
                RCv=rcv,
            )
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore
            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )

    @property
    def nic_v1(self) -> None:
        pass

    @nic_v1.setter
    def nic_v1(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        if self.nic_s is None:
            return
        with self.lock:
            hcr, vpl = pms.adsb.nic_v1(msg, self.nic_s)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                # Horizontal Containment Radius
                HCR=hcr,
                # Vertical Protection Limit
                VPL=vpl,
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore
            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )

    @property
    def nic_v2(self) -> None:
        pass

    @nic_v2.setter
    def nic_v2(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        if self.nic_a is None or self.nic_bc is None:
            return
        with self.lock:
            hcr = pms.adsb.nic_v2(msg, self.nic_a, self.nic_bc)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                # Horizontal Containment Radius
                HCR=hcr
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore

            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )

    @property
    def nuc_r(self) -> None:
        pass

    @nuc_r.setter
    def nuc_r(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            hve, vve = pms.adsb.nuc_v(msg)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                # Horizontal Velocity Error
                HVE=hve,
                # Vertical Velocity Error
                VVE=vve,
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore
            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )

    @property
    def nac_v(self) -> None:
        pass

    @nac_v.setter
    def nac_v(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            hfm, vfm = pms.adsb.nac_v(msg)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                # Horizontal Figure of Merit for rate (GNSS)
                HFM=hfm,
                # Vertical Figure of Merit for rate (GNSS)
                VFM=vfm,
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore
            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )

    @property
    def nac_p(self) -> None:
        pass

    @nac_p.setter
    def nac_p(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            epu, vepu, nacp = pms.adsb.nac_p(msg)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                # Estimated Position Uncertainty
                EPU=epu,
                # Vertical Estimated Position Uncertainty
                VEPU=vepu,
                # Navigation Accuracy Category Position
                NACp=nacp,
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore
            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )

    @property
    def sil(self) -> None:
        pass

    @sil.setter
    def sil(self, args: tuple[datetime, str]) -> None:
        t, msg = args
        with self.lock:
            phcr, pvpl, base = pms.adsb.sil(msg, self.version)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                version=self.version,
                # Probability exceeding Horizontal Containment Radius
                pHCR=phcr,
                # Probability exceeding Vertical Protection Limit
                pVPL=pvpl,
                sil_base=base,
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}  # type: ignore
            else:
                self.cumul.append(
                    dict(  # type: ignore
                        timestamp=t, icao24=self.icao24, **current
                    )
                )


class AircraftDict(Dict[str, Aircraft]):

    lat0: float
    lon0: float
    decoder: Decoder

    def __missing__(self, key: str) -> Aircraft:
        self.decoder.on_new_aircraft(key)
        self[key] = value = Aircraft(key, self.lat0, self.lon0)
        return value

    def set_latlon(self, lat0: float, lon0: float) -> None:
        self.lat0 = lat0
        self.lon0 = lon0
        for ac in list(self.values()):
            ac.lat0 = lat0
            ac.lon0 = lon0


class DumpFormat:
    def __init__(self, template: str, sep: str = ",") -> None:

        self.template = template
        self.sep = sep
        self.cols = list(x.strip() for x in template.split(sep))
        time_gen = (i for i, elt in enumerate(self.cols) if elt == "time")
        self.time_index = next(time_gen, None)
        if self.time_index is None:
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
            float(elts[self.time_index].strip()), timezone.utc  # type: ignore
        )

    def get_msg(self, line: str) -> str:
        elts = line.split(self.sep)
        return elts[self.msg_index][self.splitmsg].strip()  # type: ignore


class ModeS_Decoder:
    """

    This class implements the full data decoding process with the `pyModeS
    library <https://mode-s.org/api/>`_

    Data is first available as a list of messages per aircraft, but can be
    exported as a :class:`~traffic.core.Traffic` object.

    .. warning::

        The :meth:`from_address`, :meth:`from_dump1090`, and :meth:`from_rtlsdr`
        classmethods start a decoding thread on the creation of the object.  The
        thread can be stopped with a ``decoder.stop()`` call.

    :param reference: A reference location must be provided to decode ground
        messages. A reference can be set as:

        - a tuple of latitude/longitude coordinates
        - a str calling for an :class:`~traffic.core.structure.Airport`
          structure

        If no reference is provided (None), then reference location (latitude=0,
        longitude=0) is considered until more data can be used to readjust the
        reference.

    """

    decode_thread: Optional[StoppableThread]
    timer_thread: Optional[StoppableThread]
    timer_functions: list[
        tuple[pd.Timestamp, pd.Timedelta, Callable[[Decoder], None]]
    ] = list()

    def __init__(
        self,
        reference: None | str | Airport | tuple[float, float] = None,
        expire_threshold: str | pd.Timedelta = pd.Timedelta("10 minutes"),
        expire_frequency: str | pd.Timedelta = pd.Timedelta("1 minute"),
    ) -> None:
        """ """
        if isinstance(reference, str):
            from ...data import airports

            reference = airports[reference]

        if reference is None:
            _log.warning(
                "No valid reference position provided. Fallback to (0, 0)"
            )
            lat0, lon0 = 0.0, 0.0
        elif isinstance(reference, Airport):
            lat0, lon0 = reference.latlon
        else:
            lat0, lon0 = reference

        self.acs: AircraftDict = AircraftDict()
        self.acs.decoder = self
        self.acs.set_latlon(lat0, lon0)

        self.decode_thread = None
        self.timer_thread = None

        self.expire_threshold = (
            expire_threshold
            if isinstance(expire_threshold, pd.Timedelta)
            else pd.Timedelta(expire_threshold)
        )
        self.expire_frequency = (
            expire_frequency
            if isinstance(expire_frequency, pd.Timedelta)
            else pd.Timedelta(expire_frequency)
        )

    @classmethod
    def on_timer(
        cls, frequency: pd.Timedelta | str
    ) -> Callable[[Callable[[Decoder], None]], Callable[[Decoder], None]]:
        now = pd.Timestamp("now", tz="utc")
        if isinstance(frequency, str):
            frequency = pd.Timedelta(frequency)

        def decorate(
            function: Callable[[Decoder], None]
        ) -> Callable[[Decoder], None]:
            _log.info(f"Schedule {function.__name__} with {frequency}")
            heapq.heappush(
                cls.timer_functions,
                (now + frequency, frequency, function),
            )
            return function

        return decorate

    def expire_aircraft(self) -> None:
        _log.info("Running expire_aircraft")

        now = pd.Timestamp("now", tz="utc")

        if self.decode_thread and not self.decode_thread.is_alive():
            for icao in list(self.acs):
                self.on_expire_aircraft(icao)

        # for icao, ac in self.acs.items()
        # not compatible with changes in size of the dictionary
        for icao in list(self.acs):
            ac = self.acs[icao]
            if len(ac.cumul) > 0:
                if now - ac.cumul[-1]["timestamp"] >= self.expire_threshold:
                    self.on_expire_aircraft(icao)
            else:
                flight = ac.flight
                if flight is not None:
                    if now - flight.stop >= self.expire_threshold:
                        self.on_expire_aircraft(icao)

    def on_expire_aircraft(self, icao: str) -> None:
        with self.acs[icao].lock:
            del self.acs[icao]

    def on_new_aircraft(self, icao: str) -> None:
        _log.info(f"New aircraft {icao}")

    @classmethod
    def from_file(
        cls: Type[Decoder],
        filename: str | Path,
        reference: str | Airport | tuple[float, float],
        uncertainty: bool = False,
        crc_check: bool = False,
        template: str = "time, longmsg",
        sep: str = ",",
    ) -> Decoder:
        """Decode raw messages dumped in a text file.

        The file should contain for each line at least a timestamp and an
        hexadecimal message, as a CSV-like format.

        :param filename: the path to the file containing the data

        :param reference: the reference location, as specified above

        :param uncertainty: if True, decode also `uncertainty information
            <https://mode-s.org/decode/content/ads-b/7-uncertainty.html>`_

        :param crc_check: if True, perform CRC check on messages and discard
            invalid messages. DF 4, 5, 20 and 21 messages don't have CRC so the
            parameter should be set to False if you only have those messages.

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

        with filename.open("r") as fh:
            all_lines = fh.readlines()
            decoder = cls(reference)
            decoder.process_msgs(
                list(
                    (
                        dumpformat.get_timestamp(line),
                        dumpformat.get_msg(line),
                    )
                    for line in all_lines
                ),
                uncertainty=uncertainty,
                crc_check=crc_check,
            )
            return decoder

    @classmethod
    def from_binary(
        cls: Type[Decoder],
        filename: Union[str, Path],
        reference: Union[str, Airport, tuple[float, float]],
        *,
        uncertainty: bool = False,
        crc_check: bool = True,
        time_fmt: str = "dump1090",
        time_0: Optional[datetime] = None,
        redefine_mag: int = 10,
        fh: Optional[TextIO] = None,
    ) -> Decoder:

        decoder = cls(reference)
        redefine_freq = 2**redefine_mag - 1
        decode_time_here = decode_time.get(time_fmt, decode_time_default)

        def next_in_binary(filename: Union[str, Path]) -> Iterator[bytes]:
            with Path(filename).open("rb") as fh:
                while True:
                    get = fh.read()
                    if len(get) == 0:
                        return
                    yield get

        # We don't know the size of the binary so tqdm.rich does not work
        from tqdm.autonotebook import tqdm

        for i, bin_msg in tqdm(
            enumerate(next_beast_msg(next_in_binary(filename)))
        ):

            if len(bin_msg) < 23:
                continue

            msg = "".join(["{:02x}".format(t) for t in bin_msg])

            now = decode_time_here(msg, time_0)

            if fh is not None:
                fh.write("{},{}\n".format(now.timestamp(), msg))

            if i & redefine_freq == redefine_freq:
                decoder.redefine_reference(now)

            decoder.process(
                now,
                msg[18:],
                uncertainty=uncertainty,
                crc_check=crc_check,
            )

        return decoder

    @classmethod
    def from_rtlsdr(
        cls: Type[Decoder],
        reference: Union[str, Airport, tuple[float, float]],
        file_pattern: str = "~/ADSB_EHS_RAW_%Y%m%d_rtlsdr.csv",
        uncertainty: bool = False,
    ) -> Decoder:  # coverage: ignore
        """Decode raw messages dumped from a RTL-SDR receiver.

        :param reference: the reference location, as specified above
        :param file_pattern: the filename where to dump received hexadecimal
            messages

            Timestamp format specifiers are accepted.

            | Default value: ``"~/ADSB_EHS_RAW_%Y%m%d_rtlsdr.csv"``
            | (The ``~`` character gets expanded as your home directory)

        :param uncertainty: if True, decode also `uncertainty information
            <https://mode-s.org/decode/content/ads-b/7-uncertainty.html>`_

        .. warning::

            This method requires the `pyrtlsdr
            <https://github.com/roger-/pyrtlsdr>`_ optional dependency.

        """

        from .rtlsdr import MyRtlReader

        decoder = cls(reference)

        # dump file
        now = datetime.now(timezone.utc)
        filename = now.strftime(file_pattern)
        today = os.path.expanduser(filename)
        fh = open(today, "a", 1)

        rtlsdr = MyRtlReader(decoder, fh, uncertainty=uncertainty)
        decoder.decode_thread = StoppableThread(target=rtlsdr.run)
        decoder.decode_thread.start()
        return decoder

    @classmethod
    def from_socket(
        cls: Type[Decoder],
        s: socket.socket,
        reference: Union[str, Airport, tuple[float, float]],
        *,
        uncertainty: bool,
        crc_check: bool = True,
        time_fmt: str = "default",
        time_0: Optional[datetime] = None,
        redefine_mag: int = 7,
        fh: Optional[TextIO] = None,
    ) -> Decoder:  # coverage: ignore

        decoder = cls(reference)
        redefine_freq = 2**redefine_mag - 1
        decode_time_here = decode_time.get(time_fmt, decode_time_default)

        def next_in_tcp_socket() -> Iterator[bytes]:
            while True:
                data = s.recv(2048)
                if (
                    decoder.decode_thread is None
                    or decoder.decode_thread.to_be_stopped()
                    or len(data) == 0  # connection dropped
                ):
                    _log.warning("Connection dropped or decoder stopped")
                    s.close()
                    decoder.stop()
                    return
                yield data

        def next_in_udp_socket() -> Iterator[bytes]:
            while True:
                if (
                    decoder.decode_thread is None
                    or decoder.decode_thread.to_be_stopped()
                ):
                    s.close()
                    _log.warning("getting out of UDP socket")
                    return
                data, _addr = s.recvfrom(1024)
                yield data

        next_in_socket = {
            socket.SOCK_STREAM: next_in_tcp_socket,
            socket.SOCK_DGRAM: next_in_udp_socket,
        }

        def decode() -> None:
            for i, bin_msg in enumerate(
                next_beast_msg(next_in_socket[s.type]())
            ):

                msg = "".join(["{:02x}".format(t) for t in bin_msg])

                # Timestamp decoding
                now = decode_time_here(msg, time_0)

                if fh is not None:
                    fh.write("{},{}\n".format(now.timestamp(), msg))

                if len(bin_msg) < 23:
                    continue

                if (
                    time_fmt != "radarcape"
                    and i & redefine_freq == redefine_freq
                ):
                    decoder.redefine_reference(now)

                decoder.process(
                    now,
                    msg[18:],
                    uncertainty=uncertainty,
                    crc_check=crc_check,
                )

        def timer() -> None:
            assert decoder.decode_thread is not None

            # This one is automatically added
            cls.on_timer(decoder.expire_frequency)(cls.expire_aircraft)

            # if the decoder is not alive, finish expiring aircraft
            while decoder.decode_thread.is_alive():
                now = pd.Timestamp("now", tz="utc")
                t, delta, operation = heapq.heappop(cls.timer_functions)

                if now < t:
                    wait = t - now
                    time.sleep(wait.total_seconds())

                now = pd.Timestamp("now", tz="utc")
                operation(decoder)
                _log.info(f"Schedule {operation.__name__} at {now + delta}")
                heapq.heappush(
                    cls.timer_functions, (now + delta, delta, operation)
                )

        decoder.decode_thread = StoppableThread(target=decode)
        decoder.decode_thread.start()
        decoder.timer_thread = StoppableThread(target=timer)
        decoder.timer_thread.start()
        return decoder

    def stop(self) -> None:
        if self.decode_thread is not None and self.decode_thread.is_alive():
            self.decode_thread.stop()
            self.timer_thread.join()

    def __del__(self) -> None:
        self.stop()

    @classmethod
    def from_dump1090(
        cls: Type[Decoder],
        reference: Union[str, Airport, tuple[float, float]],
        file_pattern: str = "~/ADSB_EHS_RAW_%Y%m%d_dump1090.csv",
        uncertainty: bool = False,
    ) -> Decoder:  # coverage: ignore
        """Decode raw messages dumped from `dump1090
        <https://github.com/MalcolmRobb/dump1090/>`_

        :param reference: the reference location, as specified above
        :param file_pattern: the filename where to dump received hexadecimal
            messages

            Timestamp format specifiers are accepted.

            | Default value: ``"~/ADSB_EHS_RAW_%Y%m%d_dump1090.csv"``
            | (The ``~`` character gets expanded as your home directory)

        :param uncertainty: if True, decode also `uncertainty information
            <https://mode-s.org/decode/content/ads-b/7-uncertainty.html>`_

        .. warning::

            dump1090 must be run the ``--net`` option.

        """
        now = datetime.now(timezone.utc)
        filename = now.strftime(file_pattern)
        today = os.path.expanduser(filename)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", 30005))
        fh = open(today, "a", 1)
        return cls.from_socket(
            s,
            reference,
            uncertainty=uncertainty,
            time_fmt="dump1090",
            time_0=now,
            fh=fh,
        )

    @classmethod
    def from_address(
        cls: Type[Decoder],
        host: str,
        port: int,
        reference: Union[str, Airport, tuple[float, float]],
        file_pattern: str = "~/ADSB_EHS_RAW_%Y%m%d_tcp.csv",
        time_fmt: str = "radarcape",
        uncertainty: bool = False,
        tcp: bool = True,
    ) -> Decoder:  # coverage: ignore
        """Decode raw messages transmitted over a TCP or UDP network.

        The file should contain for each line at least a timestamp and an
        hexadecimal message, as a CSV-like format.

        :param host: the IP address of the host to connect
        :param port: the port of the host to connect
        :param reference: the reference location, as specified above
        :param file_pattern: the filename where to dump received hexadecimal
            messages

            Timestamp format specifiers are accepted.

            | Default value: ``"~/ADSB_EHS_RAW_%Y%m%d_tcp.csv"``
            | (The ``~`` character gets expanded as your home directory)

        :param time_fmt: (default: ``"radarcape"``)

            - if set to ``"radarcape"``, timestamp information included in the
              long message is assumed to be GPS adjusted and is therefore used
              to adjust the timestamp of each message;
            - if set to ``"dump1090"``, clock information provided by the
              software is used to adjust the timestamp of each message, but the
              first timestamp is based on the computer clock;
            - if set to ``"default"``, the timestamp from the computer is used
              for all decoded messages

        :param uncertainty: if True, decode also `uncertainty information
            <https://mode-s.org/decode/content/ads-b/7-uncertainty.html>`_

        """
        now = datetime.now(timezone.utc)
        filename = now.strftime(file_pattern)
        today = os.path.expanduser(filename)
        if tcp:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((host, port))
        fh = open(today, "a", 1)
        return cls.from_socket(
            s, reference, uncertainty=uncertainty, time_fmt=time_fmt, fh=fh
        )

    def redefine_reference(self, time: datetime) -> None:
        pos = list(
            (ac.lat, ac.lon)
            for ac in list(self.acs.values())
            if ac.alt is not None
            and ac.alt < 5000
            and ac.tpos is not None
            and (time - ac.tpos).total_seconds() < 20 * 60
        )
        n = len(pos)
        if n > 0:
            sum_lat = cast(float, sum(a[0] for a in pos))
            sum_lon = cast(float, sum(a[1] for a in pos))
            self.acs.set_latlon(sum_lat / n, sum_lon / n)

    def process_msgs(
        self,
        msgs: Iterable[tuple[datetime, str]],
        uncertainty: bool = False,
        crc_check: bool = True,
    ) -> None:

        for i, (t, msg) in tqdm(enumerate(msgs), total=sum(1 for _ in msgs)):
            if i & 127 == 127:
                self.redefine_reference(t)
            self.process(t, msg, uncertainty=uncertainty, crc_check=crc_check)

    def process(
        self,
        time: datetime,
        msg: str,
        *args: Any,
        uncertainty: bool = False,
        crc_check: bool = True,
        spd: Optional[float] = None,
        trk: Optional[float] = None,
        alt: Optional[float] = None,
    ) -> None:

        ac: Aircraft

        if len(msg) != 28:
            return

        df = pms.df(msg)

        if df == 4 or df == 20:
            icao = pms.icao(msg)
            if isinstance(icao, bytes):
                icao = icao.decode()
            if crc_check and icao.lower() not in self.acs:
                return
            ac = self.acs[icao.lower()]
            ac.altcode = time, msg  # type: ignore

        if df == 5 or df == 21:
            icao = pms.icao(msg)
            if isinstance(icao, bytes):
                icao = icao.decode()
            if crc_check and icao.lower() not in self.acs:
                return
            ac = self.acs[icao.lower()]
            ac.idcode = time, msg  # type: ignore

        if df == 11:
            # CRC check is valid for DF 11, 17 and 18
            # We use DF 11 messages to create aircraft when they come with no
            # ADS-B data
            if crc_check and pms.crc(msg, encode=False) != 0:
                return
            icao = pms.icao(msg)
            if isinstance(icao, bytes):
                icao = icao.decode()
            ac = self.acs[icao.lower()]

        if df == 17 or df == 18:  # ADS-B

            if crc_check and pms.crc(msg, encode=False) != 0:
                return

            tc = pms.adsb.typecode(msg)
            icao = pms.icao(msg)

            # before it's fixed in pyModeS release...
            if isinstance(icao, bytes):
                icao = icao.decode()

            ac = self.acs[icao.lower()]

            if 1 <= tc <= 4:
                ac.callsign = time, msg  # type: ignore

            if 5 <= tc <= 8:
                ac.surface = time, msg  # type: ignore

            if tc == 19:
                ac.speed = time, msg  # type: ignore

            if 9 <= tc <= 18:
                # This is barometric altitude
                ac.position = time, msg  # type: ignore

            if 20 <= tc <= 22:
                # Only GNSS altitude
                pass

            if not uncertainty:
                return

            if 9 <= tc <= 18:
                ac.nic_bc = pms.adsb.nic_b(msg)

            if (5 <= tc <= 8) or (9 <= tc <= 18) or (20 <= tc <= 22):
                ac.nuc_p = time, msg  # type: ignore
                if ac.version == 1:
                    ac.nic_v1 = time, msg  # type: ignore
                elif ac.version == 2:
                    ac.nic_v2 = time, msg  # type: ignore

            if tc == 19:
                ac.nuc_r = time, msg  # type: ignore
                if ac.version in [1, 2]:
                    ac.nac_v = time, msg  # type: ignore

            if tc == 29:
                ac.sil = time, msg  # type: ignore
                ac.nac_p = time, msg  # type: ignore

            if tc == 31:
                ac.version = pms.adsb.version(msg)
                ac.sil = time, msg  # type: ignore
                ac.nac_p = time, msg  # type: ignore

                if ac.version == 1:
                    ac.nic_s = pms.adsb.nic_s(msg)
                elif ac.version == 2:
                    ac.nic_a, ac.nic_bc = pms.adsb.nic_a_c(msg)

        elif df == 20 or df == 21:

            bds = pms.bds.infer(msg)
            icao = pms.icao(msg)
            if isinstance(icao, bytes):
                icao = icao.decode()
            ac = self.acs[icao.lower()]

            if bds == "BDS20":
                ac.bds20 = time, msg  # type: ignore
                return

            if bds == "BDS40":
                ac.bds40 = time, msg  # type: ignore
                return

            if bds == "BDS44":
                ac.bds44 = time, msg  # type: ignore
                return

            if bds == "BDS45":
                ac.bds45 = time, msg  # type: ignore
                return

            if bds == "BDS50,BDS60":
                if spd is not None and trk is not None and alt is not None:
                    bds = pms.bds.is50or60(msg, spd, trk, alt)
                elif (
                    ac.spd is not None
                    and ac.trk is not None
                    and ac.alt is not None
                ):
                    bds = pms.bds.is50or60(msg, ac.spd, ac.trk, ac.alt)
                else:
                    return
                # do not return!

            if bds == "BDS50":
                ac.bds50 = time, msg  # type: ignore
                return

            if bds == "BDS60":
                ac.bds60 = time, msg  # type: ignore
                return

    @property
    def aircraft(self) -> list[Dict[str, Any]]:
        return sorted(
            (
                dict(
                    icao24=key,
                    callsign=ac.callsign,
                    length=(
                        (len(ac.cumul) + len(ac._flight))
                        if ac._flight is not None
                        else len(ac.cumul)
                    ),
                    position=ac.lat is not None,
                    data=ac,
                )
                # avoid dictionary change size during iteration
                for (key, ac) in list(self.acs.items())
            ),
            key=itemgetter("length"),
            reverse=True,
        )

    @property
    def traffic(self) -> None | Traffic:
        """
        :return: All decoded data is converted into a
            :class:`~traffic.core.Traffic` object.

        """
        try:
            return Traffic.from_flights(
                self[elt["icao24"]] for elt in self.aircraft
            )
        except ValueError as e:
            _log.warning(e)
            return None

    def __getitem__(self, icao: str) -> Optional[Flight]:
        with self.acs[icao].lock:
            ac = self.acs[icao]
        return ac.flight
