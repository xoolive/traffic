# flake8: noqa
"""
This module implements functions for aeronautics, conversions between:

- altitude, temperature, pressure, density in the International Standard
  Atmosphere model
- computed air speed (CAS), true air speed (TAS), equivalent air speed (EAS),
  Mach number

All physical quantities assume **standard units** (SI), angles are all in
degrees.

"""

from typing import Tuple, cast

import numpy as np
import numpy.typing as npt

Numeric = npt.NDArray[np.float64]

# -- Constants Aeronautics --

kts = 0.514444  # m/s  of 1 knot
ft = 0.3048  # m    of 1 foot
fpm = ft / 60.0  # feet per minute
inch = 0.0254  # m    of 1 inch
sqft = 0.09290304  # 1sqft
nm = 1852.0  # m    of 1 nautical mile
lbs = 0.453592  # kg   of 1 pound mass
g0 = 9.80665  # m/s2    Sea level gravity constant
R = 287.05287  # Used in wikipedia table: checked with 11000 m
p0 = 101325.0  # Pa     Sea level pressure ISA
rho0 = 1.225  # kg/m3  Sea level density ISA
T0 = 288.15  # K   Sea level temperature ISA
Tstrat = 216.65  # K Stratosphere temperature (until alt=22km)
gamma = 1.40  # cp/cv for air
gamma1 = 0.2  # (gamma-1)/2 for air
gamma2 = 3.5  # gamma/(gamma-1) for air
beta = -0.0065  # [K/m] ISA temp gradient below tropopause
Rearth = 6371000.0  # m  Average earth radius
a0 = np.sqrt(gamma * R * T0)  # sea level speed of sound ISA

# -- Vectorized aero functions --


def vatmos(h: Numeric) -> Tuple[Numeric, Numeric, Numeric]:  # h in m
    """International Standard Atmosphere calculator

    :param h:  altitude in meters 0.0 < h < 84852.
        (will be clipped when outside range, integer input allowed)

    :return:
        - the pressure (in Pa)
        - the air density :math:`\\rho` (in kg/m3)
        - the temperature (in K)
    """
    # Temp
    T = vtemp(h)

    # Density
    rhotrop = 1.225 * (T / 288.15) ** 4.256848030018761
    dhstrat = np.maximum(0.0, h - 11000.0)
    rho = rhotrop * np.exp(-dhstrat / 6341.552161)  # = *g0/(287.05*216.65))

    # Pressure
    p = rho * R * T

    return p, rho, T


def vtemp(h: Numeric) -> Numeric:  # h [m]
    """Temperature only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the temperature (in K)

    """
    T = np.maximum(288.15 - 0.0065 * h, Tstrat)
    return T


# Atmos wrappings:
def vpressure(h: Numeric) -> Numeric:  # h [m]
    """Pressure only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the pressure, in Pa

    """
    p, r, T = vatmos(h)
    return p


def vdensity(h: Numeric) -> Numeric:  # air density at given altitude h [m]
    """Density only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the density :math:`\\rho`, in kg/m3

    """
    p, r, T = vatmos(h)
    return r


def vvsound(h: Numeric) -> Numeric:  # Speed of sound for given altitude h [m]
    """Speed of sound in ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the speed of sound :math:`a`, in m/s

    """
    T = vtemp(h)
    a = np.sqrt(gamma * R * T)
    return a  # type: ignore


# -- Speed conversions --


def vtas2mach(tas: Numeric, h: Numeric) -> Numeric:
    """
    :param tas: True Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Mach number
    """
    a = vvsound(h)
    M: Numeric = tas / a
    return M


def vmach2tas(M: Numeric, h: Numeric) -> Numeric:
    """
    :param M: Mach number
    :param h: altitude, (in m)

    :return: True Air Speed, (in m/s)
    """
    a = vvsound(h)
    tas: Numeric = M * a
    return tas


def veas2tas(eas: Numeric, h: Numeric) -> Numeric:
    """
    :param eas: Equivalent Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: True Air Speed, (in m/s)
    """
    rho = vdensity(h)
    tas = eas * np.sqrt(rho0 / rho)
    return tas  # type: ignore


def vtas2eas(tas: Numeric, h: Numeric) -> Numeric:
    """
    :param tas: True Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Equivalent Air Speed, (in m/s)
    """
    rho = vdensity(h)
    eas = tas * np.sqrt(rho / rho0)
    return eas  # type: ignore


def vcas2tas(cas: Numeric, h: Numeric) -> Numeric:
    """
    :param cas: Computed Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: True Air Speed, (in m/s)
    """
    p, rho, T = vatmos(h)
    qdyn = p0 * ((1.0 + rho0 * cas * cas / (7.0 * p0)) ** 3.5 - 1.0)
    tas = np.sqrt(7.0 * p / rho * ((1.0 + qdyn / p) ** (2.0 / 7.0) - 1.0))

    # cope with negative speed
    tas = np.where(cas < 0, -1 * tas, tas)
    return tas


def vtas2cas(tas: Numeric, h: Numeric) -> Numeric:
    """
    :param tas: True Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Computed Air Speed, (in m/s)
    """
    p, rho, T = vatmos(h)
    qdyn = p * ((1.0 + rho * tas * tas / (7.0 * p)) ** 3.5 - 1.0)
    cas = np.sqrt(7.0 * p0 / rho0 * ((qdyn / p0 + 1.0) ** (2.0 / 7.0) - 1.0))

    # cope with negative speed
    cas = np.where(tas < 0, -1 * cas, cas)
    return cas  # type: ignore


def vmach2cas(M: Numeric, h: Numeric) -> Numeric:
    """
    :param M: Mach number
    :param h: altitude, (in m)

    :return: Computed Air Speed, (in m/s)
    """
    tas = vmach2tas(M, h)
    cas = vtas2cas(tas, h)
    return cas


def vcas2mach(cas: Numeric, h: Numeric) -> Numeric:
    """
    :param cas: Computed Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Mach number
    """
    tas = vcas2tas(cas, h)
    M = vtas2mach(tas, h)
    return M


def vcasormach(spd: Numeric, h: Numeric) -> Tuple[Numeric, Numeric, Numeric]:
    ismach = np.logical_and(0.1 < spd, spd < 1)
    tas = np.where(ismach, vmach2tas(spd, h), vcas2tas(spd, h))
    cas = np.where(ismach, vtas2cas(tas, h), spd)
    m = np.where(ismach, spd, vtas2mach(tas, h))
    return tas, cas, m


def vcasormach2tas(spd: Numeric, h: Numeric) -> Numeric:
    tas = np.where(np.abs(spd) < 1, vmach2tas(spd, h), vcas2tas(spd, h))
    return tas


# -- Scalar aero functions --


def atmos(h: float) -> Tuple[float, float, float]:
    """International Standard Atmosphere calculator

    :param h:  altitude in meters :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return:
        - the pressure (in Pa)
        - the air density :math:`\\rho` (in kg/m3)
        - the temperature (in K)

    """

    # Constants

    # Base values and gradient in table from hand-out
    # (but corrected to avoid small discontinuities at borders of layers)
    h0 = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 86852.0]

    p0 = [
        101325.0,  # Sea level
        22631.7009099,  # 11 km
        5474.71768857,  # 20 km
        867.974468302,  # 32 km
        110.898214043,  # 47 km
        66.939,  # 51 km
        3.9564,
    ]  # 71 km

    T0 = [
        288.15,  # Sea level
        216.65,  # 11 km
        216.65,  # 20 km
        228.65,  # 32 km
        270.65,  # 47 km
        270.65,  # 51 km
        214.65,
    ]  # 71 km

    # a = lapse rate (temp gradient)
    # integer 0 indicates isothermic layer!
    a = [
        -0.0065,  # 0-11 km
        0,  # 11-20 km
        0.001,  # 20-32 km
        0.0028,  # 32-47 km
        0,  # 47-51 km
        -0.0028,  # 51-71 km
        -0.002,
    ]  # 71-   km

    # Clip altitude to maximum!
    h = max(0.0, min(float(h), h0[-1]))

    # Find correct layer
    i = 0
    while h > h0[i + 1] and i < len(h0) - 2:
        i = i + 1

    # Calculate if sothermic layer
    if a[i] == 0:
        T = T0[i]
        p = p0[i] * np.exp(-g0 / (R * T) * (h - h0[i]))
        rho = p / (R * T)

    # Calculate for temperature gradient
    else:
        T = T0[i] + a[i] * (h - h0[i])
        p = p0[i] * ((T / T0[i]) ** (-g0 / (a[i] * R)))
        rho = p / (R * T)

    return cast(float, p), rho, T


def temp(h: float) -> float:
    """Temperature only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the temperature (in K)

    """

    # Base values and gradient in table from hand-out
    # (but corrected to avoid small discontinuities at borders of layers)
    h0 = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 86852.0]

    T0 = [
        288.15,  # Sea level
        216.65,  # 11 km
        216.65,  # 20 km
        228.65,  # 32 km
        270.65,  # 47 km
        270.65,  # 51 km
        214.65,
    ]  # 71 km

    # a = lapse rate (temp gradient)
    # integer 0 indicates isothermic layer!
    a = [
        -0.0065,  # 0-11 km
        0,  # 11-20 km
        0.001,  # 20-32 km
        0.0028,  # 32-47 km
        0,  # 47-51 km
        -0.0028,  # 51-71 km
        -0.002,
    ]  # 71-   km

    # Clip altitude to maximum!
    h = max(0.0, min(float(h), h0[-1]))

    # Find correct layer
    i = 0
    while h > h0[i + 1] and i < len(h0) - 2:
        i = i + 1

    # Calculate if sothermic layer
    if a[i] == 0:
        T = T0[i]

    # Calculate for temperature gradient
    else:
        T = T0[i] + a[i] * (h - h0[i])

    return T


# Atmos wrappings:
def pressure(h: float) -> float:  # h [m]
    """Pressure only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the pressure, in Pa

    """
    p, r, T = atmos(h)
    return p


def density(h: float) -> float:  # air density at given altitude h [m]
    """Density only version of ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the density :math:`\\rho`, in kg/m3

    """
    p, r, T = atmos(h)
    return r


def vsound(h: float) -> float:  # Speed of sound for given altitude h [m]
    """Speed of sound in ISA atmosphere

    :param h: the altitude in meters, :math:`0 < h < 84852`
        (will be clipped when outside range, integer input allowed)

    :return: the speed of sound :math:`a`, in m/s

    """
    T = temp(h)
    a = np.sqrt(gamma * R * T)
    return a  # type: ignore


# -- Speed conversions --


def tas2mach(tas: float, h: float) -> float:
    """
    :param tas: True Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Mach number
    """
    a = vsound(h)
    M = tas / a
    return M


def mach2tas(M: float, h: float) -> float:
    """
    :param M: Mach number
    :param h: altitude, (in m)

    :return: True Air Speed, (in m/s)
    """
    a = vsound(h)
    tas = M * a
    return tas


def eas2tas(eas: float, h: float) -> float:
    """
    :param eas: Equivalent Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: True Air Speed, (in m/s)
    """
    rho = density(h)
    tas = eas * np.sqrt(rho0 / rho)
    return tas  # type: ignore


def tas2eas(tas: float, h: float) -> float:
    """
    :param tas: True Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Equivalent Air Speed, (in m/s)
    """
    rho = density(h)
    eas = tas * np.sqrt(rho / rho0)
    return eas  # type: ignore


def cas2tas(cas: float, h: float) -> float:
    """
    :param cas: Computed Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: True Air Speed, (in m/s)
    """
    p, rho, T = atmos(h)
    qdyn = p0 * ((1.0 + rho0 * cas * cas / (7.0 * p0)) ** 3.5 - 1.0)
    tas = np.sqrt(7.0 * p / rho * ((1.0 + qdyn / p) ** (2.0 / 7.0) - 1.0))
    tas = -1 * tas if cas < 0 else tas
    return tas  # type: ignore


def tas2cas(tas: float, h: float) -> float:
    """
    :param tas: True Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Computed Air Speed, (in m/s)
    """
    p, rho, T = atmos(h)
    qdyn = p * ((1.0 + rho * tas * tas / (7.0 * p)) ** 3.5 - 1.0)
    cas = np.sqrt(7.0 * p0 / rho0 * ((qdyn / p0 + 1.0) ** (2.0 / 7.0) - 1.0))
    cas = -1 * cas if tas < 0 else cas
    return cas  # type: ignore


def mach2cas(M: float, h: float) -> float:
    """
    :param M: Mach number
    :param h: altitude, (in m)

    :return: Computed Air Speed, (in m/s)
    """
    tas = mach2tas(M, h)
    cas = tas2cas(tas, h)
    return cas


def cas2mach(cas: float, h: float) -> float:
    """
    :param cas: Computed Air Speed, (in m/s)
    :param h: altitude, (in m)

    :return: Mach number
    """
    tas = cas2tas(cas, h)
    M = tas2mach(tas, h)
    return M


def casormach(spd: float, h: float) -> float:
    if 0.1 < spd < 1:
        # Interpret spd as Mach number
        tas = mach2tas(spd, h)
        cas = mach2cas(spd, h)
        m = spd
    else:
        # Interpret spd as CAS
        tas = cas2tas(spd, h)
        cas = spd
        m = cas2mach(spd, h)
    return tas, cas, m  # type: ignore


def casormach2tas(spd: float, h: float) -> float:
    if 0.1 < spd < 1:
        # Interpret spd as Mach number
        tas = mach2tas(spd, h)
    else:
        # Interpret spd as CAS
        tas = cas2tas(spd, h)
    return tas


def metres_to_feet_rounded(metres: float) -> int:
    """
    Converts metres to feet.
    Returns feet as rounded integer.
    """
    return int(round(metres / ft))


def metric_spd_to_knots_rounded(speed: float) -> int:
    """
    Converts speed in m/s to knots.
    Returns knots as rounded integer.
    """
    return int(round(speed / kts))
