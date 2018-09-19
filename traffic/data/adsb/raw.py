from operator import itemgetter
from collections import UserDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

import pandas as pd
import pyModeS as pms
from tqdm.autonotebook import tqdm
from traffic.core import Flight, Traffic
from traffic.data.basic.airport import Airport
from traffic.data import airports


class Aircraft(object):
    def __init__(self, icao24: str, lat0: float, lon0: float) -> None:
        self.icao24 = icao24
        self._callsign: Optional[str] = None
        self.cumul: List[Dict] = []

        self.t0: Optional[datetime] = None
        self.t1: Optional[datetime] = None
        self.tpos: Optional[datetime] = None

        self.m0: Optional[str] = None
        self.m1: Optional[str] = None

        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.alt: Optional[float] = None

        self.lat0: float = lat0
        self.lon0: float = lon0

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
            return
        if (spd is None) or (trk is None):
            return

        self.cumul.append(
            dict(
                timestamp=t,
                icao24=self.icao24,
                groundspeed=spd,
                track_angle=trk,
                vertical_speed=roc,
            )
        )

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

            self.cumul.append(
                dict(
                    timestamp=t,
                    icao24=self.icao24,
                    latitude=self.lat,
                    longitude=self.lon,
                    altitude=self.alt,
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
        self.cumul.append(
            dict(
                timestamp=t,
                icao24=self.icao24,
                latitude=self.lat,
                longitude=self.lon,
            )
        )

    @property
    def bs20(self):
        pass

    @bs20.setter
    def bs20(self, args):
        t, msg = args
        callsign = pms.adsb.callsign(msg).strip("_")
        if callsign == "":
            return
        self._callsign = callsign
        self.cumul.append(
            dict(timestamp=t, icao24=self.icao24, callsign=self._callsign)
        )

    @property
    def bs40(self):
        pass

    @bs40.setter
    def bs40(self, args):
        t, msg = args
        self.cumul.append(
            dict(
                timestamp=t,
                icao24=self.icao24,
                alt_fms=pms.commb.alt40fms(msg),
                alt_mcp=pms.commb.alt40mcp(msg),
                p_baro=pms.commb.p40baro(msg),
            )
        )

    @property
    def bs44(self):
        pass

    @bs44.setter
    def bs44(self, args):
        t, msg = args
        wind = pms.commb.wind44(msg)
        wind = wind if wind is not None else (None, None)
        self.cumul.append(
            dict(
                timestamp=t,
                icao24=self.icao24,
                humidity=pms.commb.hum44(msg),
                pression=pms.commb.p44(msg),
                temperature=pms.commb.temp44(msg),
                windspeed=wind[0],
                winddirection=wind[1],
            )
        )

    @property
    def bs50(self):
        pass

    @bs50.setter
    def bs50(self, args):
        t, msg = args
        self.cumul.append(
            dict(
                timestamp=t,
                icao24=self.icao24,
                gs=pms.commb.gs50(msg),
                roll=pms.commb.roll50(msg),
                tas=pms.commb.tas50(msg),
                track=pms.commb.trk50(msg),
                track_rate=pms.commb.rtrk50(msg),
            )
        )

    @property
    def bs60(self):
        pass

    @bs60.setter
    def bs60(self, args):
        t, msg = args
        self.cumul.append(
            dict(
                timestamp=t,
                icao24=self.icao24,
                ias=pms.commb.ias60(msg),
                heading=pms.commb.hdg60(msg),
                mach=pms.commb.mach60(msg),
                vrbaro=pms.commb.vr60baro(msg),
                vrins=pms.commb.vr60ins(msg),
            )
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
    def __init__(
        self, reference: Union[str, Airport, Tuple[float, float]]
    ) -> None:
        if isinstance(reference, str):
            reference = airports[reference]
        if isinstance(reference, Airport):
            lat0, lon0 = reference.lat, reference.lon
        else:
            lat0, lon0 = cast(Tuple[float, float], reference)

        self.acs = AircraftDict()
        self.acs.set_latlon(lat0, lon0)

    @classmethod
    def from_file(
        cls,
        filename: Union[str, Path],
        reference: Union[str, Airport, Tuple[float, float]],
    ):

        if isinstance(filename, str):
            filename = Path(filename)

        with filename.open("r") as fh:
            all_lines = fh.readlines()
            decoder = cls(reference)
            decoder.process(
                list(
                    (
                        datetime.fromtimestamp(
                            float(line.strip().split(",")[0])
                        ),
                        cast(str, line.strip().split(",")[1][18:]),
                    )
                    for line in all_lines
                )
            )
            return decoder

    def process(self, msgs: Iterable[Tuple[datetime, str]]):

        for i, (time, msg) in tqdm(enumerate(msgs), total=sum(1 for _ in msgs)):

            if i & 127 == 127:
                # reset the reference lat/lon
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

            if int(pms.crc(msg, encode=False), 2) != 0:
                continue

            icao = pms.icao(msg)
            if icao is None:
                print(icao)
                continue

            ac = self.acs[icao.lower()]
            df = pms.df(msg)

            if df == 17 or df == 18:
                # ADS-B
                tc = pms.adsb.typecode(msg)

                if 1 <= tc <= 4:
                    ac.callsign = time, msg

                if 5 <= tc <= 8:
                    ac.surface = time, msg

                if tc == 19:
                    ac.speed = time, msg

                if 9 <= tc <= 18:
                    ac.position = time, msg

                # if 9 <= tc <= 18:
                #     ac["nic_bc"] = pms.adsb.nic_b(msg)

                # if (5 <= tc <= 8) or (9 <= tc <= 18) or (20 <= tc <= 22):
                #     ac["HPL"], ac["RCu"], ac["RCv"] = pms.adsb.nuc_p(msg)

                #     if (ac["ver"] == 1) and ("nic_s" in ac.keys()):
                #         ac["Rc"], ac["VPL"] = pms.adsb.nic_v1(msg, ac["nic_s"])
                #     elif (
                #         (ac["ver"] == 2)
                #         and ("nic_a" in ac.keys())
                #         and ("nic_bc" in ac.keys())
                #     ):
                #         ac["Rc"] = pms.adsb.nic_v2(msg, ac["nic_a"], ac["nic_bc"])

                # if tc == 19:
                #     ac["HVE"], ac["VVE"] = pms.adsb.nuc_v(msg)
                #     if ac["ver"] in [1, 2]:
                #         ac["EPU"], ac["VEPU"] = pms.adsb.nac_v(msg)

                # if tc == 29:
                #     ac["PE_RCu"], ac["PE_VPL"], ac["base"] = pms.adsb.sil(
                #         msg, ac["ver"]
                #     )
                #     ac["HFOMr"], ac["VFOMr"] = pms.adsb.nac_p(msg)

                # if tc == 31:
                #     ac["ver"] = pms.adsb.version(msg)
                #     ac["HFOMr"], ac["VFOMr"] = pms.adsb.nac_p(msg)
                #     ac["PE_RCu"], ac["PE_VPL"], ac["sil_base"] = pms.adsb.sil(
                #         msg, ac["ver"]
                #     )

                #     if ac["ver"] == 1:
                #         ac["nic_s"] = pms.adsb.nic_s(msg)
                #     elif ac["ver"] == 2:
                #         ac["nic_a"], ac["nic_bc"] = pms.adsb.nic_a_c(msg)

            elif df == 20 or df == 21:

                bds = pms.bds.infer(msg)

                if bds == "BDS20":
                    ac.bs20 = time, msg

                if bds == "BDS40":
                    ac.bs40 = time, msg

                if bds == "BDS44":
                    ac.bs40 = time, msg

                if bds == "BDS50":
                    ac.bds50 = time, msg

                elif bds == "BDS60":
                    ac.bds60 = time, msg

    @property
    def aircraft(self):
        return sorted(
            (
                dict(
                    icao24=key,
                    callsign=ac.callsign,
                    length=len(ac.cumul),
                    position=ac.lat is not None,
                    data=ac,
                )
                for (key, ac) in self.acs.items()
                if len(ac.cumul) > 0 and ac.callsign is not None
            ),
            key=itemgetter("length"),
            reverse=True,
        )

    @property
    def traffic(self):
        return Traffic.from_flights(
            [self[elt["icao24"]] for elt in self.aircraft]
        )

    def __getitem__(self, icao):
        df = pd.DataFrame.from_records(self.acs[icao].cumul)
        return Flight(
            df.assign(
                callsign=df.callsign.replace("", None)
                .fillna(method="ffill")
                .fillna(method="bfill")
            )
        )
