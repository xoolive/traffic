# fmt: off

import logging
import os
import signal
import socket
import threading
from collections import UserDict
from datetime import datetime, timedelta, timezone
from operator import itemgetter
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator,
    List, Optional, TextIO, Tuple, Union
)

import pandas as pd
import pyModeS as pms
from tqdm.autonotebook import tqdm

from ...core import Flight, Traffic
from ...data.basic.airports import Airport

# fmt: on


def next_msg(chunk_it: Iterator[bytes]) -> Iterator[bytes]:
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
            if data[1] == 0x33:
                yield data[:23]
                data = data[23:]
                continue
            elif data[1] == 0x32:
                data = data[16:]
                continue
            elif data[1] == 0x31:
                data = data[11:]
                continue
            elif data[1] == 0x34:
                data = data[23:]
                continue
            else:
                data = data[1:]


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
    now = now.replace(hour=0, minute=0, second=0, microsecond=0)
    now += timedelta(seconds=secs, microseconds=nanos / 1000)
    return now


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


decode_time: Dict[str, Callable[[str, Optional[datetime]], datetime]] = {
    "radarcape": decode_time_radarcape,
    "dump1090": decode_time_dump1090,
    "default": decode_time_default,
}


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the to_be_stopped() condition."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.daemon = True  # is it redundant?
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def to_be_stopped(self) -> bool:
        return self._stop_event.is_set()


class Aircraft(object):
    def __init__(self, icao24: str, lat0: float, lon0: float) -> None:
        self.icao24 = icao24
        self._callsign: Optional[str] = None
        self._flight: Optional[Flight] = None
        self.cumul: List[Dict] = []

        self.t0: Optional[datetime] = None
        self.t1: Optional[datetime] = None
        self.tpos: Optional[datetime] = None

        self.m0: Optional[str] = None
        self.m1: Optional[str] = None

        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.alt: Optional[float] = None
        self.trk: Optional[float] = None
        self.spd: Optional[float] = None

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
            df = pd.concat([self._flight.data, df], sort=False)
            if self.version is not None:
                # remove columns added by nuc_p, nuc_r
                if "HPL" in df.columns:
                    df = df.drop(columns=["HPL", "RCu", "RCv"])
                if "HVE" in df.columns:
                    df = df.drop(columns=["HVE", "VVE"])

        if len(df) == 0:
            return None

        self._flight = Flight(
            df.assign(
                callsign=df.callsign.replace("", None)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
        )

        return self._flight

    @property
    def callsign(self):
        return self._callsign

    @callsign.setter
    def callsign(self, args):
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
    def speed(self):
        pass

    @speed.setter
    def speed(self, args):
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
    def position(self):
        pass

    @position.setter
    def position(self, args):
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
    def surface(self):
        pass

    @surface.setter
    def surface(self, args):
        t, msg = args
        self.lat, self.lon = pms.adsb.surface_position_with_ref(
            msg, self.lat0, self.lon0
        )
        with self.lock:
            self.cumul.append(
                dict(
                    timestamp=t,
                    icao24=self.icao24,
                    latitude=self.lat,
                    longitude=self.lon,
                    onground=True,
                )
            )

    @property
    def altcode(self):
        pass

    @altcode.setter
    def altcode(self, args):
        t, msg = args
        from pyModeS.c_common import hex2bin

        if set(hex2bin(msg)[19:32]) in [{"0"}, {"1"}]:
            return
        self.alt = pms.common.altcode(msg)
        with self.lock:
            self.cumul.append(
                dict(timestamp=t, icao24=self.icao24, altitude=self.alt)
            )

    @property
    def idcode(self):
        pass

    @idcode.setter
    def idcode(self, args):
        t, msg = args
        from pyModeS.c_common import hex2bin

        if set(hex2bin(msg)[19:32]) in [{"0"}, {"1"}]:
            return
        idcode = pms.common.idcode(msg)
        with self.lock:
            self.cumul.append(
                dict(timestamp=t, icao24=self.icao24, squawk=idcode,)
            )

    @property
    def bds20(self):
        pass

    @bds20.setter
    def bds20(self, args):
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
                self.cumul[-1] = dict(**last_entry, callsign=self._callsign)
            else:
                self.cumul.append(
                    dict(
                        timestamp=t, icao24=self.icao24, callsign=self._callsign
                    )
                )

    @property
    def bds40(self):
        pass

    @bds40.setter
    def bds40(self, args):
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {
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
    def bds44(self):
        pass

    @bds44.setter
    def bds44(self, args):
        t, msg = args
        wind = pms.commb.wind44(msg)
        wind = wind if wind is not None else (None, None)
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF 5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {
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
    def bds45(self):
        pass

    @bds45.setter
    def bds45(self, args):
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF 5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {
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
    def bds50(self):
        pass

    @bds50.setter
    def bds50(self, args):
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {
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
    def bds60(self):
        pass

    @bds60.setter
    def bds60(self, args):
        t, msg = args
        with self.lock:
            # in case altitude was already included from altcode (DF 4 or 20)
            # or squawk from idcode (DF5 or 21)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {
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
    def nuc_p(self):
        pass

    @nuc_p.setter
    def nuc_p(self, args):
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
                self.cumul[-1] = {**last_entry, **current}
            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )

    @property
    def nic_v1(self):
        pass

    @nic_v1.setter
    def nic_v1(self, args):
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
                self.cumul[-1] = {**last_entry, **current}
            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )

    @property
    def nic_v2(self):
        pass

    @nic_v2.setter
    def nic_v2(self, args):
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
                self.cumul[-1] = {**last_entry, **current}

            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )

    @property
    def nuc_r(self):
        pass

    @nuc_r.setter
    def nuc_r(self, args):
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
                self.cumul[-1] = {**last_entry, **current}
            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )

    @property
    def nac_v(self):
        pass

    @nac_v.setter
    def nac_v(self, args):
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
                self.cumul[-1] = {**last_entry, **current}
            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )

    @property
    def nac_p(self):
        pass

    @nac_p.setter
    def nac_p(self, args):
        t, msg = args
        with self.lock:
            epu, vepu = pms.adsb.nac_p(msg)
            last_entry = self.cumul[-1] if len(self.cumul) > 0 else None
            current = dict(
                # Estimated Position Uncertainty
                EPU=epu,
                # Vertical Estimated Position Uncertainty
                VEPU=vepu,
            )
            if last_entry is not None and last_entry["timestamp"] == t:
                self.cumul[-1] = {**last_entry, **current}
            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )

    @property
    def sil(self):
        pass

    @sil.setter
    def sil(self, args):
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
                self.cumul[-1] = {**last_entry, **current}
            else:
                self.cumul.append(
                    dict(timestamp=t, icao24=self.icao24, **current)
                )


class AircraftDict(UserDict):

    lat0: float
    lon0: float

    def __missing__(self, key):
        self[key] = value = Aircraft(key, self.lat0, self.lon0)
        return value

    def set_latlon(self, lat0, lon0):
        self.lat0 = lat0
        self.lon0 = lon0
        for ac in self.values():
            ac.lat0 = lat0
            ac.lon0 = lon0


class Decoder:
    thread: Optional[StoppableThread]

    def __init__(
        self, reference: Union[None, str, Airport, Tuple[float, float]] = None
    ) -> None:
        if isinstance(reference, str):
            from ...data import airports

            reference = airports[reference]

        if reference is None:
            logging.warning(
                "No valid reference position provided. Fallback to (0, 0)"
            )
            lat0, lon0 = 0.0, 0.0
        elif isinstance(reference, Airport):
            lat0, lon0 = reference.latlon
        else:
            lat0, lon0 = reference

        self.acs: AircraftDict = AircraftDict()
        self.acs.set_latlon(lat0, lon0)
        self.thread = None

    @classmethod
    def from_file(
        cls,
        filename: Union[str, Path],
        reference: Union[str, Airport, Tuple[float, float]],
        uncertainty: bool = False,
    ) -> "Decoder":

        if isinstance(filename, str):
            filename = Path(filename)

        with filename.open("r") as fh:
            all_lines = fh.readlines()
            decoder = cls(reference)
            decoder.process_msgs(
                list(
                    (
                        datetime.fromtimestamp(
                            float(line.strip().split(",")[0]), timezone.utc
                        ),
                        line.strip().split(",")[1][18:],
                    )
                    for line in all_lines
                ),
                uncertainty=uncertainty,
            )
            return decoder

    @classmethod
    def from_binary(
        cls,
        filename: Union[str, Path],
        reference: Union[str, Airport, Tuple[float, float]],
        *,
        uncertainty: bool = False,
        time_fmt: str = "dump1090",
        time_0: Optional[datetime] = None,
        redefine_mag: int = 10,
        fh: Optional[TextIO] = None,
    ):

        decoder = cls(reference)
        redefine_freq = 2 ** redefine_mag - 1
        decode_time_here = decode_time.get(time_fmt, decode_time_default)

        def next_in_binary(filename: Union[str, Path]) -> Iterator[bytes]:
            with Path(filename).open("rb") as fh:
                while True:
                    get = fh.read()
                    if len(get) == 0:
                        return
                    yield get

        for i, bin_msg in tqdm(enumerate(next_msg(next_in_binary(filename)))):

            if len(bin_msg) < 23:
                continue

            msg = "".join(["{:02x}".format(t) for t in bin_msg])

            now = decode_time_here(msg, time_0)

            if fh is not None:
                fh.write("{},{}\n".format(now.timestamp(), msg))

            if i & redefine_freq == redefine_freq:
                decoder.redefine_reference(now)

            decoder.process(now, msg[18:], uncertainty=uncertainty)

        return decoder

    @classmethod
    def from_rtlsdr(
        cls,
        reference: Union[str, Airport, Tuple[float, float]],
        file_pattern: str = "~/ADSB_EHS_RAW_%Y%m%d_dump1090.csv",
        uncertainty: bool = False,
    ) -> "Decoder":  # coverage: ignore

        from .rtlsdr import MyRtlReader

        decoder = cls(reference)

        # dump file
        now = datetime.now(timezone.utc)
        filename = now.strftime(file_pattern)
        today = os.path.expanduser(filename)
        fh = open(today, "a", 1)

        rtlsdr = MyRtlReader(decoder, fh, uncertainty=uncertainty)
        decoder.thread = StoppableThread(target=rtlsdr.run)
        signal.signal(signal.SIGINT, rtlsdr.stop)
        decoder.thread.start()
        return decoder

    @classmethod
    def from_socket(
        cls,
        socket: socket.socket,
        reference: Union[str, Airport, Tuple[float, float]],
        *,
        uncertainty: bool,
        time_fmt: str = "default",
        time_0: Optional[datetime] = None,
        redefine_mag: int = 7,
        fh: Optional[TextIO] = None,
    ) -> "Decoder":  # coverage: ignore

        decoder = cls(reference)
        redefine_freq = 2 ** redefine_mag - 1
        decode_time_here = decode_time.get(time_fmt, decode_time_default)

        def next_in_socket() -> Iterator[bytes]:
            while True:
                if decoder.thread is None or decoder.thread.to_be_stopped():
                    socket.close()
                    return
                yield socket.recv(2048)

        def decode():
            for i, bin_msg in enumerate(next_msg(next_in_socket())):

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

                decoder.process(now, msg[18:], uncertainty=uncertainty)

        decoder.thread = StoppableThread(target=decode)
        decoder.thread.start()
        return decoder

    def stop(self):
        if self.thread is not None and self.thread.is_alive():
            self.thread.stop()
            self.thread.join()

    def __del__(self):
        self.stop()

    @classmethod
    def from_dump1090(
        cls,
        reference: Union[str, Airport, Tuple[float, float]],
        file_pattern: str = "~/ADSB_EHS_RAW_%Y%m%d_dump1090.csv",
        uncertainty: bool = False,
    ) -> "Decoder":  # coverage: ignore
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
        cls,
        host: str,
        port: int,
        reference: Union[str, Airport, Tuple[float, float]],
        file_pattern: str = "~/ADSB_EHS_RAW_%Y%m%d_tcp.csv",
        uncertainty: bool = False,
    ) -> "Decoder":  # coverage: ignore
        now = datetime.now(timezone.utc)
        filename = now.strftime(file_pattern)
        today = os.path.expanduser(filename)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        fh = open(today, "a", 1)
        return cls.from_socket(
            s, reference, uncertainty=uncertainty, time_fmt="radarcape", fh=fh
        )

    def redefine_reference(self, time: datetime) -> None:
        pos = list(
            (ac.lat, ac.lon)
            for ac in self.acs.values()
            if ac.alt is not None
            and ac.alt < 5000
            and ac.tpos is not None
            and (time - ac.tpos).total_seconds() < 20 * 60
        )
        n = len(pos)
        if n > 0:
            self.acs.set_latlon(
                sum(a[0] for a in pos) / n, sum(a[1] for a in pos) / n
            )

    def process_msgs(
        self, msgs: Iterable[Tuple[datetime, str]], uncertainty: bool = False
    ) -> None:

        for i, (time, msg) in tqdm(enumerate(msgs), total=sum(1 for _ in msgs)):
            if i & 127 == 127:
                self.redefine_reference(time)
            self.process(time, msg, uncertainty=uncertainty)

    def process(
        self,
        time: datetime,
        msg: str,
        *args,
        uncertainty: bool = False,
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
            ac = self.acs[icao.lower()]
            ac.altcode = time, msg

        if df == 5 or df == 21:
            icao = pms.icao(msg)
            if isinstance(icao, bytes):
                icao = icao.decode()
            ac = self.acs[icao.lower()]
            ac.idcode = time, msg

        if df == 17 or df == 18:  # ADS-B

            if pms.crc(msg, encode=False) != 0:
                return

            tc = pms.adsb.typecode(msg)
            icao = pms.icao(msg)

            # before it's fixed in pyModeS release...
            if isinstance(icao, bytes):
                icao = icao.decode()

            ac = self.acs[icao.lower()]

            if 1 <= tc <= 4:
                ac.callsign = time, msg

            if 5 <= tc <= 8:
                ac.surface = time, msg

            if tc == 19:
                ac.speed = time, msg

            if 9 <= tc <= 18:
                # This is barometric altitude
                ac.position = time, msg

            if 20 <= tc <= 22:
                # Only GNSS altitude
                pass

            if not uncertainty:
                return

            if 9 <= tc <= 18:
                ac.nic_bc = pms.adsb.nic_b(msg)

            if (5 <= tc <= 8) or (9 <= tc <= 18) or (20 <= tc <= 22):
                ac.nuc_p = time, msg
                if ac.version == 1:
                    ac.nic_v1 = time, msg
                elif ac.version == 2:
                    ac.nic_v2 = time, msg

            if tc == 19:
                ac.nuc_r = time, msg
                if ac.version in [1, 2]:
                    ac.nac_v = time, msg

            if tc == 29:
                ac.sil = time, msg
                ac.nac_p = time, msg

            if tc == 31:
                ac.version = pms.adsb.version(msg)
                ac.sil = time, msg
                ac.nac_p = time, msg

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
                ac.bds20 = time, msg
                return

            if bds == "BDS40":
                ac.bds40 = time, msg
                return

            if bds == "BDS44":
                ac.bds44 = time, msg
                return

            if bds == "BDS45":
                ac.bds45 = time, msg
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
                ac.bds50 = time, msg
                return

            if bds == "BDS60":
                ac.bds60 = time, msg
                return

    @property
    def aircraft(self) -> List[Dict[str, Any]]:
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
                for (key, ac) in self.acs.items()
                if ac.callsign is not None
            ),
            key=itemgetter("length"),
            reverse=True,
        )

    @property
    def traffic(self) -> Optional[Traffic]:
        try:
            return Traffic.from_flights(
                self[elt["icao24"]] for elt in self.aircraft
            )
        except ValueError:
            return None

    def __getitem__(self, icao: str) -> Optional[Flight]:
        return self.acs[icao].flight
