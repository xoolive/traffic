from __future__ import annotations

from itertools import count

from impunity import impunity
from pitot import aero

import numpy as np

from ...core import Flight
from ...core import types as tt


class FuelflowEstimation:
    """Estimates the fuel flow with OpenAP.

    The OpenAP model is based on the aircraft type (actually, the most probable
    engine type) and on three features commonly available in ADS-B data:

    - altitude (in ft),
    - vertical rate (in ft/min), and
    - speed (in kts), in order of priority, ``TAS`` (true air speed), ``CAS``
      (computed air speed, used to compute TAS) and ``groundspeed``, if no air
      speed is available.

    :param initial_mass: by default (None), 90% of the maximum take-off weight.
      You can also pass a value to initialise the mass.
      When ``initial_mass > 1``, the mass is in kg. When ``initial_mass <= 1``,
      it represents the fraction of the maximum take-off weight.

    :param typecode: by default (None), use the typecode column if available,
      the provided aircraft database to infer the typecode based
      on the ``icao24``. Ignored if the engine parameter is not None.

    :param engine: by default (None), use the default engine associated with
      the aircraft type.

    :return: the same instance enriched with three extra features: the mass,
      the fuel flow (in kg/s) and the total burnt fuel (in kg).

    """

    def __init__(
        self,
        initial_mass: None | str | float = None,
        typecode: None | str = None,
        engine: None | str = None,
    ):
        self.initial_mass = initial_mass
        self.typecode = typecode
        self.engine = engine

    @impunity(ignore_warnings=True)
    def estimate(self, flight: Flight) -> Flight:
        import openap

        typecode = (
            self.typecode if self.typecode is not None else flight.typecode
        )

        if typecode is None:
            return flight

        available_aircraft = openap.prop.available_aircraft(use_synonym=True)
        if typecode.lower() not in available_aircraft:
            return flight

        ac = openap.prop.aircraft(typecode)

        update_mass = True

        if self.initial_mass is not None:
            if not isinstance(self.initial_mass, str):
                if self.initial_mass <= 1:
                    # treat as percentage of max weight
                    mass = (
                        self.initial_mass
                        * ac["limits"]["MTOW"]
                        * np.ones_like(flight.data.altitude)
                    )
                else:
                    mass = self.initial_mass * np.ones_like(
                        flight.data.altitude
                    )
        elif isinstance(self.initial_mass, str) and hasattr(
            flight.data, self.initial_mass
        ):
            mass = flight.data[self.initial_mass].values
            update_mass = False
        else:
            mass = (
                0.9 * ac["limits"]["MTOW"] * np.ones_like(flight.data.altitude)
            )

        fuelflow = openap.FuelFlow(typecode, eng=self.engine, use_synonym=True)

        dt: tt.seconds_array = (
            flight.data.timestamp.diff().dt.total_seconds().bfill().values
        )

        TAS: tt.speed_array = flight.data.get("TAS", None)
        altitude: tt.altitude_array = flight.data.altitude.to_numpy()
        vertical_rate: tt.vertical_rate_array = (
            flight.data.vertical_rate.to_numpy()
        )

        if TAS is None:
            CAS: tt.speed_array = flight.data.get("CAS", None)
            if CAS is not None:
                TAS = aero.cas2tas(CAS.to_numpy(), altitude)
        else:
            TAS = TAS.to_numpy()  # type: ignore

        if TAS is None:
            TAS = flight.data.groundspeed.to_numpy()  # unit: knots

        FF = []
        Fuel = []
        for i, tas, alt, vs, dti in zip(
            count(1), TAS, altitude, vertical_rate, dt
        ):
            ff = fuelflow.enroute(mass=mass[i - 1], tas=tas, alt=alt, vs=vs)
            if update_mass:
                mass[i:] -= ff * dti if ff == ff else 0
            FF.append(ff)
            Fuel.append(mass[0] - mass[i - 1])

        return flight.assign(mass=mass, fuel=Fuel, fuelflow=FF, dt=dt)


class PollutantEstimation:
    """Estimates the fuel flow with OpenAP.

    The estimation method is based on the :class:FuelflowEstimation which is
    also called on the same instance.

    :return: the same instance with new columns for various pollutants,
      including H20, HC, CO, CO2, NOx and SOx are added to the data frame.
    """

    def __init__(
        self,
        initial_mass: None | float = None,
        typecode: None | str = None,
        engine: None | str = None,
    ):
        self.initial_mass = initial_mass
        self.typecode = typecode
        self.engine = engine

    @impunity
    def estimate(self, flight: Flight) -> Flight:
        import openap

        typecode = (
            self.typecode if self.typecode is not None else flight.typecode
        )

        if typecode is None:
            return flight

        available_aircraft = openap.prop.available_aircraft(use_synonym=True)
        if typecode.lower() not in available_aircraft:
            return flight

        if "fuelflow" not in flight.data.columns:
            flight = FuelflowEstimation(
                initial_mass=self.initial_mass,
                typecode=self.typecode,
                engine=self.engine,
            ).estimate(flight)

        if "fuelflow" not in flight.data.columns:
            # fuel flow cannot be computed
            return flight

        dt: tt.seconds_array = flight.data.dt
        TAS: tt.speed_array = flight.data.get("TAS", None)
        altitude: tt.altitude_array = flight.data.altitude.to_numpy()

        if TAS is None:
            CAS: tt.speed_array = flight.data.get("CAS", None)
            if CAS is not None:
                CAS = CAS.to_numpy()
                TAS = aero.cas2tas(CAS, altitude)
        else:
            TAS = TAS.to_numpy()  # type: ignore

        if TAS is None:
            TAS = flight.data.groundspeed.to_numpy()

        emission = openap.Emission(
            ac=typecode, eng=self.engine, use_synonym=True
        )

        NOx = emission.nox(flight.data.fuelflow, tas=TAS, alt=altitude)
        CO = emission.co(flight.data.fuelflow, tas=TAS, alt=altitude)
        HC = emission.hc(flight.data.fuelflow, tas=TAS, alt=altitude)
        CO2 = emission.co2(flight.data.fuelflow)
        H2O = emission.h2o(flight.data.fuelflow)
        SOx = emission.sox(flight.data.fuelflow)

        return flight.assign(
            **{
                "NOx": (NOx.cumsum() * dt * 1e-3),
                "CO": (CO.cumsum() * dt * 1e-3),
                "HC": (HC.cumsum() * dt * 1e-3),
                "CO2": (CO2.cumsum() * dt * 1e-3),
                "H2O": (H2O.cumsum() * dt * 1e-3),
                "SOx": (SOx.cumsum() * dt * 1e-3),
            }
        )
