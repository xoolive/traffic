from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Union, cast

from impunity import impunity
from pitot import aero
from typing_extensions import Annotated

import numpy as np
import numpy.typing as npt

from ..core import types as tt

if TYPE_CHECKING:
    from ..core import Flight


@impunity
def compute_path_angle(
    vertical_rate: Annotated[npt.NDArray[np.float64], "m/s"],
    speed: Annotated[npt.NDArray[np.float64], "m/s"],
) -> tt.angle_array:
    return np.degrees(np.arctan2(vertical_rate, speed))  # type: ignore


class OpenAP:
    @impunity
    def phases(self, twindow: int = 60) -> "Flight":
        """Assign a flight phase to each timestamp of a flight
        using OpenAP phase detection fuzzy logic method.
        """

        from openap.phase import FlightPhase

        # The following cast secures the typing
        self = cast("Flight", self)

        altitude: tt.altitude_array = self.data.altitude.values
        groundspeed: tt.speed_array = self.data.groundspeed.values
        vertical_rate: tt.vertical_rate_array = self.data.vertical_rate.values

        fp = FlightPhase()
        fp.set_trajectory(
            (self.data.timestamp.values - np.datetime64("1970-01-01"))
            / np.timedelta64(1, "s"),
            altitude,
            groundspeed,
            vertical_rate,
        )
        return self.assign(phase=fp.phaselabel(twindow=twindow)).assign(
            phase=lambda df: df.phase.str.replace("GND", "GROUND")
            .str.replace("CL", "CLIMB")
            .str.replace("DE", "DESCENT")
            .str.replace("CR", "CRUISE")
            .str.replace("LVL", "LEVEL")
        )

    @impunity(ignore_warnings=True)
    def fuelflow(
        self,
        initial_mass: Union[None, str, float] = None,
        typecode: Union[None, str] = None,
        engine: Union[None, str] = None,
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
          weight. You can also pass a value to initialise the mass.
          When ``initial_mass > 1``, the mass is in kg. When
          ``initial_mass <= 1``, it represents the fraction of the maximum
          take-off weight.

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

        available_aircraft = openap.prop.available_aircraft(use_synonym=True)
        if actype.lower() not in available_aircraft:
            return self

        ac = openap.prop.aircraft(actype)

        update_mass = True

        if initial_mass is not None:
            if not isinstance(initial_mass, str):
                if initial_mass <= 1:
                    # treat as percentage of max weight
                    mass = (
                        initial_mass
                        * ac["limits"]["MTOW"]
                        * np.ones_like(self.data.altitude)
                    )
                else:
                    mass = initial_mass * np.ones_like(self.data.altitude)
        elif isinstance(initial_mass, str) and hasattr(self.data, initial_mass):
            mass = self.data[initial_mass].values
            update_mass = False
        else:
            mass = 0.9 * ac["limits"]["MTOW"] * np.ones_like(self.data.altitude)

        fuelflow = openap.FuelFlow(actype, eng=engine, use_synonym=True)

        dt: tt.seconds_array = (
            self.data.timestamp.diff().dt.total_seconds().bfill().values
        )

        TAS: tt.speed_array = self.data.get("TAS", None)
        altitude: tt.altitude_array = self.data.altitude
        vertical_rate: tt.vertical_rate_array = self.data.vertical_rate

        if TAS is None:
            CAS: tt.speed_array = self.data.get("CAS", None)
            if CAS is not None:
                TAS = aero.cas2tas(CAS, altitude)

        if TAS is None:
            TAS = self.data.groundspeed  # unit: knots

        path_angle: tt.angle_array = compute_path_angle(vertical_rate, TAS)

        FF = []
        Fuel = []
        for i, tas, alt, pa, dti in zip(
            count(1), TAS, altitude, path_angle, dt
        ):
            ff = fuelflow.enroute(
                mass=mass[i - 1], tas=tas, alt=alt, path_angle=pa
            )
            if update_mass:
                mass[i:] -= ff * dti if ff == ff else 0
            FF.append(ff)
            Fuel.append(mass[0] - mass[i - 1])

        return self.assign(mass=mass, fuel=Fuel, fuelflow=FF, dt=dt)

    @impunity
    def emission(
        self, mass: Union[None, float] = None, engine: Union[None, str] = None
    ) -> "Flight":
        import openap

        # The following cast secures the typing
        self = cast("Flight", self)

        actype = self.typecode

        if actype is None:
            return self

        available_aircraft = openap.prop.available_aircraft(use_synonym=True)
        if actype.lower() not in available_aircraft:
            return self

        if "fuelflow" not in self.data.columns:
            self = self.fuelflow(mass)

        if "fuelflow" not in self.data.columns:
            # fuel flow cannot be computed
            return self

        dt: tt.seconds_array = self.data.dt
        TAS: tt.speed_array = self.data.get("TAS", None)
        altitude: tt.altitude_array = self.data.altitude

        if TAS is None:
            CAS: tt.speed_array = self.data.get("CAS", None)
            if CAS is not None:
                TAS = aero.cas2tas(CAS, altitude)

        if TAS is None:
            TAS = self.data.groundspeed

        emission = openap.Emission(ac=actype, eng=engine, use_synonym=True)

        NOx = emission.nox(self.data.fuelflow, tas=TAS, alt=altitude)
        CO = emission.co(self.data.fuelflow, tas=TAS, alt=altitude)
        HC = emission.hc(self.data.fuelflow, tas=TAS, alt=altitude)
        CO2 = emission.co2(self.data.fuelflow)
        H2O = emission.h2o(self.data.fuelflow)
        SOx = emission.sox(self.data.fuelflow)

        return self.assign(
            **{
                "NOx": (NOx.cumsum() * dt * 1e-3),
                "CO": (CO.cumsum() * dt * 1e-3),
                "HC": (HC.cumsum() * dt * 1e-3),
                "CO2": (CO2.cumsum() * dt * 1e-3),
                "H2O": (H2O.cumsum() * dt * 1e-3),
                "SOx": (SOx.cumsum() * dt * 1e-3),
            }
        )
