from __future__ import annotations
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from ..core import Flight  # noqa: F401


class OpenAP:
    def phases(self, twindow: int = 60) -> "Flight":
        """Assign a flight phase to each timestamp of a flight
        using OpenAP phase detection fuzzy logic method.
        """

        from openap.phase import FlightPhase

        # The following cast secures the typing
        self = cast("Flight", self)

        fp = FlightPhase()
        fp.set_trajectory(
            (self.data.timestamp.values - np.datetime64("1970-01-01"))
            / np.timedelta64(1, "s"),
            self.data.altitude.values,
            self.data.groundspeed.values,
            self.data.vertical_rate.values,
        )
        return self.assign(phase=fp.phaselabel(twindow=twindow)).assign(
            phase=lambda df: df.phase.str.replace("GND", "GROUND")
            .str.replace("CL", "CLIMB")
            .str.replace("DE", "DESCENT")
            .str.replace("CR", "CRUISE")
            .str.replace("LVL", "LEVEL")
        )

    def fuelflow(self, mass: None | float = None) -> "Flight":

        import openap

        # The following cast secures the typing
        self = cast("Flight", self)

        actype = self.typecode

        if actype != actype or (  # None or nan
            actype is not None
            and actype.lower() not in openap.prop.available_aircraft()
        ):
            return self

        ac = openap.prop.aircraft(actype)

        if mass is None:
            mass = 0.9 * ac["limits"]["MTOW"]

        fuelflow = openap.FuelFlow(actype, use_synonym=True)

        dt = self.data.timestamp.diff().dt.total_seconds().bfill().values

        TAS = self.data.get("TAS", self.data.groundspeed)
        VR = self.data.vertical_rate
        ALT = self.data.altitude
        PA = np.degrees(np.arctan2(VR * 0.00508, TAS * 0.51445))

        Mass = []
        FF = []
        Fuel = []
        mass0 = mass
        for (tas, alt, pa, dt) in zip(TAS, ALT, PA, dt):
            ff = fuelflow.enroute(mass=mass, tas=tas, alt=alt, path_angle=pa)
            mass -= ff * dt
            assert mass is not None
            Mass.append(round(mass, 1))
            FF.append(round(float(ff), 4))
            Fuel.append(mass0 - mass)

        return self.assign(mass=Mass, fuel=Fuel, fuelflow=FF, dt=dt)

    def emission(self, mass: None | float = None) -> "Flight":

        import openap

        if "fuelflow" not in self.data.columns:  # type: ignore
            self = self.fuelflow(mass)

        # The following cast secures the typing
        self = cast("Flight", self)

        if "fuelflow" not in self.data.columns:
            # fuel flow cannot be computed
            return self

        emission = openap.Emission(ac=self.typecode, use_synonym=True)

        NOx = emission.nox(
            self.data.fuelflow,
            tas=self.data.get("TAS", self.data.groundspeed),
            alt=self.data.altitude,
        )
        CO = emission.co(
            self.data.fuelflow,
            tas=self.data.get("TAS", self.data.groundspeed),
            alt=self.data.altitude,
        )
        HC = emission.hc(
            self.data.fuelflow,
            tas=self.data.get("TAS", self.data.groundspeed),
            alt=self.data.altitude,
        )
        CO2 = emission.co2(self.data.fuelflow)
        H2O = emission.h2o(self.data.fuelflow)
        SOx = emission.sox(self.data.fuelflow)

        return self.assign(
            **{
                "nox": (NOx.cumsum() * self.data.dt * 1e-3).round(2),
                "co": (CO.cumsum() * self.data.dt * 1e-3).round(2),
                "hc": (HC.cumsum() * self.data.dt * 1e-3).round(2),
                "co2": (CO2.cumsum() * self.data.dt * 1e-3).round(2),
                "h2o": (H2O.cumsum() * self.data.dt * 1e-3).round(2),
                "sox": (SOx.cumsum() * self.data.dt * 1e-3).round(2),
            }
        )
