from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, cast

import numpy as np

from ..core import aero

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

    def fuelflow(
        self,
        initial_mass: None | str | float = None,
        typecode: None | str = None,
        engine: None | str = None,
    ) -> "Flight":
        """Estimates the fuel flow with OpenAP.

        The OpenAP model is based on the aircraft type (actually, the most
        probable engine type) and on three features commonly available in ADS-B
        data:

        - altitude (in ft),
        - vertical rate (in ft/min), and
        - speed (in kts), in order of priority, ``TAS`` (true air speed),
          ``CAS`` (computed air speed, used to compute TAS) and ``groundspeed``,
          if no air speed is available.

        :param initial_mass: by default (None), 90% of the maximum take-off
          weight. If an existing feature name is passed, use it to initialise
          the mass. You can also pass a value in kg.

        :param typecode: by default (None), use the typecode column if
          available, the provided aircraft database to infer the typecode based
          on the ``icao24``. Ignored if the engine parameter is not None.

        :param engine: by default (None), use the default engine associated with
          the aircraft type.

        :return: the same instance enriched with three extra features: the mass,
          the fuel flow (in kg/s) and the total burnt fuel (in kg).

        """

        import openap

        # The following cast secures the typing
        self = cast("Flight", self)

        actype = typecode if typecode is not None else self.typecode

        if actype is None:
            return self

        if actype.lower() not in openap.prop.available_aircraft():
            return self

        ac = openap.prop.aircraft(actype)

        update_mass = True
        if initial_mass is not None:
            mass = initial_mass * np.ones_like(self.data.altitude)
        elif isinstance(initial_mass, str) and hasattr(self.data, initial_mass):
            mass = self.data[initial_mass].values
            update_mass = False
        else:
            mass = 0.9 * ac["limits"]["MTOW"] * np.ones_like(self.data.altitude)

        fuelflow = openap.FuelFlow(actype, eng=engine, use_synonym=True)

        dt = self.data.timestamp.diff().dt.total_seconds().bfill().values

        TAS = self.data.get("TAS", None)
        if TAS is None:
            CAS = self.data.get("CAS", None)
            if CAS is not None:
                TAS = aero.vcas2tas(
                    CAS * aero.kts, self.data.altitude * aero.ft
                )

        if TAS is None:
            TAS = self.data.groundspeed

        VR = self.data.vertical_rate
        ALT = self.data.altitude
        PA = np.degrees(np.arctan2(VR * 0.00508, TAS * 0.51445))

        FF = []
        Fuel = []
        for (i, tas, alt, pa, dt) in zip(count(1), TAS, ALT, PA, dt):
            ff = fuelflow.enroute(
                mass=mass[i - 1], tas=tas, alt=alt, path_angle=pa
            )
            if update_mass:
                mass[i:] -= ff * dt if ff == ff else 0
            FF.append(round(float(ff), 4))
            Fuel.append(mass[0] - mass[i - 1])

        return self.assign(mass=mass, fuel=Fuel, fuelflow=FF, dt=dt)

    def emission(
        self, mass: None | float = None, engine: None | str = None
    ) -> "Flight":

        import openap

        # The following cast secures the typing
        self = cast("Flight", self)

        actype = self.typecode

        if actype is None:
            return self

        if actype.lower() not in openap.prop.available_aircraft():
            return self

        if "fuelflow" not in self.data.columns:
            self = self.fuelflow(mass)

        if "fuelflow" not in self.data.columns:
            # fuel flow cannot be computed
            return self

        TAS = self.data.get("TAS", None)
        if TAS is None:
            CAS = self.data.get("CAS", None)
            if CAS is not None:
                TAS = aero.vcas2tas(
                    CAS * aero.kts, self.data.altitude * aero.ft
                )

        if TAS is None:
            TAS = self.data.groundspeed

        emission = openap.Emission(ac=actype, eng=engine, use_synonym=True)

        NOx = emission.nox(
            self.data.fuelflow,
            tas=TAS,
            alt=self.data.altitude,
        )
        CO = emission.co(
            self.data.fuelflow,
            tas=TAS,
            alt=self.data.altitude,
        )
        HC = emission.hc(
            self.data.fuelflow,
            tas=TAS,
            alt=self.data.altitude,
        )
        CO2 = emission.co2(self.data.fuelflow)
        H2O = emission.h2o(self.data.fuelflow)
        SOx = emission.sox(self.data.fuelflow)

        return self.assign(
            **{
                "NOx": (NOx.cumsum() * self.data.dt * 1e-3).round(2),
                "CO": (CO.cumsum() * self.data.dt * 1e-3).round(2),
                "HC": (HC.cumsum() * self.data.dt * 1e-3).round(2),
                "CO2": (CO2.cumsum() * self.data.dt * 1e-3).round(2),
                "H2O": (H2O.cumsum() * self.data.dt * 1e-3).round(2),
                "SOx": (SOx.cumsum() * self.data.dt * 1e-3).round(2),
            }
        )
