from io import BytesIO
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import pandas as pd

Numeric = npt.NDArray[np.float64]

anp_file_dict = {
    "aircraft": [
        "ACFT_ID",
        "Description",
        "Engine Type",
        "Number Of Engines",
        "Weight Class",
        "Owner Category",
        "Max Gross Takeoff Weight (lb)",
        "Max Gross Landing Weight (lb)",
        "Max Landing Distance (ft)",
        "Max Sea Level Static Thrust (lb)",
        "Noise Chapter",
        "NPD_ID",
        "Power Parameter",
        "Approach Spectral Class ID",
        "Departure Spectral Class ID",
        "Lateral Directivity Identifier",
    ],
    "default_weights": [
        "ACFT_ID",
        "Stage Length",
        "Weight (lb)",
    ],
    "aerodynamic_coefficients": [
        "ACFT_ID",
        "Op Type",
        "Flap_ID",
        "B",
        "C",
        "D",
        "R",
    ],
    "jet_engine_coefficients": [
        "ACFT_ID",
        "Thrust Rating",
        "E",
        "F",
        "Ga",
        "Gb",
        "H",
    ],
    "propeller_engine_coefficients": [
        "ACFT_ID",
        "Thrust Rating",
        "Propeller Efficiency",
        "Installed Net Propulsive Power (hp)",
    ],
    "default_approach_procedural_steps": [
        "ACFT_ID",
        "Profile_ID",
        "Step Number",
        "Step Type",
        "Flap_ID",
        "Start Altitude(ft)",
        "Start CAS (kt)",
        "Descent Angle (deg)",
        "Touchdown Roll (ft)",
        "Distance (ft)",
        "Start Thrust",
    ],
    "default_departure_procedural_steps": [
        "ACFT_ID",
        "Profile_ID",
        "Stage Length",
        "Step Number",
        "Step Type",
        "Thrust Rating",
        "Flap_ID",
        "End Point Altitude (ft)",
        "Rate Of Climb (ft/min)",
        "End Point CAS (kt)",
        "Accel Percentage (%)",
    ],
    "default_fixed_point_profiles": [
        "ACFT_ID",
        "Op Type",
        "Profile_ID",
        "Stage Length",
        "Point Number",
        "Distance (ft)",
        "Altitude AFE (ft)",
        "TAS (kt)",
        "Power Setting",
    ],
}


def calculate_stage_length(flight_distance: float) -> int:
    if flight_distance < 500.0:
        return 1

    if flight_distance < 1000.0:
        return 2

    if flight_distance < 1500.0:
        return 3

    if flight_distance < 2500.0:
        return 4

    if flight_distance < 3500.0:
        return 5

    if flight_distance < 4500.0:
        return 6

    if flight_distance < 5500.0:
        return 7

    if flight_distance < 6500.0:
        return 8

    return 9


class Anp(object):
    """
    Anp contains the ANP tables (keys in anp_file_dict) stored as
    separate DataFrames. The tables names, variables and values are not
    changed after loading the tables.
    The tables can be directly accessed as attributes
    (.aircraft, .aerodynamic_coefficients, ...)

    Additionally, a flap retraction schedule table is provided (used for
    thrust computations in the arrival phase with a force-balance equation)

    Some convenience methods are provided for the most common
    performance calculations with ANP data.
    """

    flap_retraction_file: Path = Path(__file__).parent / "Flap Schedules.csv"

    def __init__(
            self,
            anp_zip_path: Union[str, None] = None,
            anp_substitution_path: Union[str, None] = None,
    ):
        """
        :param anp_zip_path: Path to a zip file with the ANP tables. If
        None no ANP tables will be loaded.

        :param anp_substitution_path: Path to a local .xlsx file with the
        ANP substitution table. If None the substitution table will not be
        loaded.
        """

        if anp_zip_path is not None:
            self._anp_from_zip(anp_zip_path)

        if anp_substitution_path is not None:
            self._substitution_from_xlsx(anp_substitution_path)

        self.flap_retraction_schedule = pd.read_csv(self.flap_retraction_file)

    # -- Aircraft Properties --
    def engine_count(self, acft_id: str) -> int:
        return int(
            self.aircraft.loc[  # type: ignore
                self.aircraft["ACFT_ID"] == acft_id,  # type: ignore
                "Number Of Engines",
            ].iloc[0]
        )

    def landing_weight(self, acft_id: str, percentage: float = 1.0) -> float:
        return float(
            self.aircraft.loc[  # type: ignore
                self.aircraft["ACFT_ID"] == acft_id,  # type: ignore
                "Max Gross Landing Weight (lb)",
            ].iloc[0]
            * percentage
        )

    def takeoff_weight(
            self,
            acft_id: str,
            percentage: float = 1.0,
            stage_length: Union[None, int] = None,
            flight_distance: Union[None, float] = None,
    ) -> float:
        """
        Get the takeoff weight of an aircraft as MTOW multiplied by a percentage
        or from a stage length.

        If both stage_length and flight_distance are None,
         MTOW * percentage will be returned

        If stage_length is not None, the respective weight will be returned.

        If flight_distance is not None, the corresponding stage_length will
        be calculated and then the respective weight returned.
        """

        if stage_length is None:
            if flight_distance is None:
                return float(
                    self.aircraft.loc[  # type: ignore
                        self.aircraft["ACFT_ID"] == acft_id,  # type: ignore
                        "Max Gross Takeoff Weight (lb)",
                    ]
                    * percentage
                )
            stage_length = calculate_stage_length(flight_distance)

        df = self.default_weights  # type: ignore
        weight = float(
            df.loc[
                (
                        df["ACFT_ID"] == acft_id
                        and df["Stage Length"] == stage_length
                ),
                "Weight (lb)",
            ]
        )

        return weight * percentage

    # -- Table Filtering --
    def filter(self, by: Union[str, List[str]] = "all") -> "Anp":
        """
        Filter the anp tables so that entries which are not found
        in specific tables are removed.

        :param by: Allowed values (case insensitive) are:
            -Flap retraction: keep only entries with id found in the flap
            retractions table

            -Substitution: keep only entries with id found in the substitution
            table

            -Thrust coefficients: keep only entries with id found either in the
            jet_engine_coefficients or propeller_engine_coefficients table

            -Aerodynamic coefficients: keep only entries with id found in the
            aerodynamic_coefficients table
        """

        if isinstance(by, str):
            if by.lower() == "all":
                by = [
                    "default weights",
                    "flap retraction",
                    "substitution",
                    "thrust coefficients",
                    "aerodynamic coefficients",
                ]
            else:
                by = [by]

        by = [x.lower() for x in by]
        r = self

        if "default weights" in by:
            r._filter(set(r.default_weights["ACFT_ID"]))  # type: ignore
        if "flap retraction" in by:
            r._filter(
                set(r.flap_retraction_schedule["ACFT_ID"])  # type: ignore
            )

        if "substitution" in by:
            r._filter(set(r.substitution["ANP_PROXY"]))  # type: ignore

        if "thrust coefficients" in by:
            ids = set(
                r.jet_engine_coefficients["ACFT_ID"]  # type: ignore
            ) | set(
                r.propeller_engine_coefficients["ACFT_ID"]  # type: ignore
            )
            r._filter(ids)

        if "aerodynamic coefficients" in by:
            r._filter(
                set(r.aerodynamic_coefficients["ACFT_ID"])  # type: ignore
            )

        return r

    def _filter(self, acft_ids: Set[str]) -> None:
        for anp_file in anp_file_dict:
            df = getattr(self, anp_file)
            df = df.loc[df["ACFT_ID"].isin(acft_ids), :]
            setattr(self, anp_file, df)

        s = self.substitution  # type: ignore
        s = s.loc[s["ANP_PROXY"].isin(acft_ids)]
        self.substitution = s

        f = self.flap_retraction_schedule  # type: ignore
        f = f.loc[f["ACFT_ID"].isin(acft_ids)]
        self.flap_retraction_schedule = f

    # -- Physical Quantities --
    def thrust_force_balance(
            self,
            acft_id: str,
            w: Union[int, float, Numeric, None],
            cas: Numeric,
            accel: Numeric,
            ang: Numeric,
            p: Numeric,
            w_percentage: float = 1.0,
    ) -> Numeric:
        """
        Calculate arrival thrust for n points calculated with the force balance
        equation B-20 of Doc29

        :param acft_id: aircraft to use
        :param w: (default: None) the weight of the aircraft, either a single
        value or an array of n values.
        If None the aircraft landing weight will be used.
        :param cas: an array on n calibrated airspeeds
        :param accel: an array of n accelerations
        :param ang: an array of n climb angles
        :param p: an array of n pressures
        :param w_percentage: (default: ``1.0``) will be multiplied with w
        """

        if w is None:
            w = self.landing_weight(acft_id, w_percentage)
        else:
            w *= w_percentage

        r = self._r_coefficient(acft_id, cas)
        ang_rad = np.radians(ang)

        return np.array(
            w
            * (r * np.cos(ang_rad) + np.sin(ang_rad) + accel / 9.80665)
            / (self.engine_count(acft_id) * p / 1013.25)
        )

    def thrust_rating(
            self,
            acft_id: str,
            alt: Optional[Numeric],
            cas: Optional[Numeric],
            tas: Optional[Numeric],
            temp: Optional[Numeric],
            press: Optional[Numeric],
            cutback: Union[float, int, None] = None,
            vert_rate: Optional[Numeric] = None,
            break_temp: float = 30.0,
    ) -> Tuple[Numeric, int]:
        """
        Calculate departure thrust for n points with the thrust rating
        equations B-1, B-4 and B-5 of Doc29.

        For jet engines, alt, cas and temp are mandatory.
        For propeller engines, tas and press are mandatory.

        Either cutback or vert_rate must be passed. These parameters
        determine how the cutback point is estimated.
        - If cutback is an int, it is used as an index to the cutback point
        - If cutback is a float, the cutback index is found by searching the
        alt array (assumes an ascendily sorted array)
        - If cutback is None, vert_rate must be passed and the first local
        maxima (width of 3) is used as cutback index.

        :param acft_id: aircraft to use
        :param alt: an array on n altitudes
        :param cas: an array on n calibrated airspeeds
        :param tas: an array on n true airspeeds
        :param temp: an array of n temperatures
        :param press: an array of n pressures
        :param cutback: (default: ``1500``) the altitude at which thrust
        cutback was performed
        :param break_temp: (default: ``30``) the temperature at which the
        engine thrust output begins to sink

        :returns The calculated thrust and the index at which thrust cutback
        was estimated.
        """
        from scipy.signal import find_peaks

        high_temperature = temp[0] > break_temp if temp is not None else False

        # Find thrust cutback index
        if cutback is None:
            assert all(param is not None for param in [alt, vert_rate])

            alt_afe = alt - alt[0]  # type: ignore
            idx_500 = alt_afe.searchsorted(500)
            idx_3300 = alt_afe.searchsorted(3300, side="right")
            # TODO: Find peaks based on derivative
            idx, _ = find_peaks(
                vert_rate[idx_500:idx_3300], width=3  # type: ignore
            )

            # Defaults back to 1500ft thrust cutback if no peaks are found
            idx = idx[0] if idx.size != 0 else alt_afe.searchsorted(1500)

            # Convert back to complete dataframe index
            # and takeoff thrust precedence
            idx += idx_500 + 1
        elif isinstance(cutback, float):
            assert alt is not None

            idx = np.searchsorted(alt, cutback, side="left")
        else:
            idx = cutback

        if (
                acft_id
                in self.jet_engine_coefficients["ACFT_ID"].values
                # type: ignore
        ):
            assert all(param is not None for param in [alt, cas, temp])

            c = self.jet_engine_coefficients.loc[  # type: ignore
                self.jet_engine_coefficients["ACFT_ID"]  # type: ignore
                == acft_id
                ]

            if high_temperature and all(
                    rating in c["Thrust Rating"].values
                    for rating in ["MaxTkoffHiTemp", "MaxClimbHiTemp"]
            ):
                c_t = c.loc[c["Thrust Rating"] == "MaxTkoffHiTemp"].squeeze()
                c_c = c.loc[c["Thrust Rating"] == "MaxClimbHiTemp"].squeeze()
                high_temperature = False
            else:
                c_t = c.loc[c["Thrust Rating"] == "MaxTakeoff"].squeeze()
                c_c = c.loc[c["Thrust Rating"] == "MaxClimb"].squeeze()

            if not high_temperature:
                takeoff_thrust = (
                        c_t["E"]
                        + c_t["F"] * cas[:idx]  # type: ignore
                        + c_t["Ga"] * alt[:idx]  # type: ignore
                        + c_t["Gb"] * alt[:idx] * alt[:idx]  # type: ignore
                        + c_t["H"] * temp[:idx]  # type: ignore
                )

                climb_thrust = (
                        c_c["E"]
                        + c_c["F"] * cas[idx:]  # type: ignore
                        + c_c["Ga"] * alt[idx:]  # type: ignore
                        + c_c["Gb"] * alt[idx:] * alt[idx:]  # type: ignore
                        + c_c["H"] * temp[idx:]  # type: ignore
                )
                return np.concatenate([takeoff_thrust, climb_thrust]), idx

            else:
                takeoff_thrust = c_t["F"] * cas[:idx] + (  # type: ignore
                        c_t["E"] + c_t["H"] * break_temp
                ) * (
                                         1 - 0.006 * temp[:idx]  # type: ignore
                                 ) / (
                                         1 - 0.006 * break_temp
                                 )

                climb_thrust = c_c["F"] * cas[idx:] + (  # type: ignore
                        c_c["E"] + c_c["H"] * break_temp
                ) * (
                                       1 - 0.006 * temp[idx:]  # type: ignore
                               ) / (
                                       1 - 0.006 * break_temp
                               )
                return np.concatenate([takeoff_thrust, climb_thrust]), idx

        elif (
                acft_id in
                (self.propeller_engine_coefficients[
                    "ACFT_ID"
                ].values)  # type: ignore
        ):
            assert all(param is not None for param in [tas, press])

            c = self.propeller_engine_coefficients.loc[  # type: ignore
                self.propeller_engine_coefficients["ACFT_ID"]  # type: ignore
                == acft_id
                ]
            c_t = c.loc[c["Thrust Rating"] == "MaxTakeoff"].squeeze()
            c_c = c.loc[c["Thrust Rating"] == "MaxClimb"].squeeze()

            # Propeller thrust is very high thrust at low true air speeds
            # Values is capped at 125% of rated thrust
            max_thrust = self.aircraft.loc[
                self.aircraft["ACFT_ID"] == acft_id,
                "Max Sea Level Static Thrust (lb)"
            ].astype(float) * 1.25

            takeoff_thrust = (
                                     326
                                     * c_t["Propeller Efficiency"]
                                     * c_t[
                                         "Installed Net Propulsive Power (hp)"]
                                     / tas[:idx]  # type: ignore
                             ) / (
                                     press[:idx] / 1013.25  # type: ignore
                             )
            takeoff_thrust = np.minimum(takeoff_thrust.values,
                                        max_thrust.values)

            climb_thrust = (
                                   326
                                   * c_c["Propeller Efficiency"]
                                   * c_c["Installed Net Propulsive Power (hp)"]
                                   / tas[idx:]  # type: ignore
                           ) / (
                                   press[idx:] / 1013.25  # type: ignore
                           )
            climb_thrust = np.minimum(climb_thrust.values, max_thrust.values)

            return np.concatenate([takeoff_thrust, climb_thrust]), idx
        else:
            raise RuntimeError(
                f"Neither jet nor propeller engine "
                f"coefficients were found for aircraft {acft_id}"
            )

    def _r_coefficient(self, acft_id: str, cas: Numeric) -> Numeric:
        acft_schedule = self.flap_retraction_schedule[
            self.flap_retraction_schedule["ACFT_ID"] == acft_id
            ]
        idx = np.maximum(
            np.searchsorted(
                -acft_schedule["Start CAS (kt)"].values, -cas, side="right"
            )
            - 1,
            0,
        )
        return np.array(acft_schedule["R"].iloc[idx].cummax())

    def _get_flap_retraction_schedule(self) -> None:
        self.flap_retraction_schedule = pd.read_csv(self.flap_retraction_file)

    def _anp_from_zip(self, path: Path) -> None:
        zp = ZipFile(path)
        zp_file_list = [zip_file_info.filename for zip_file_info in zp.filelist]

        for anp_file, cols in anp_file_dict.items():
            if zp_file := next(
                    (
                            zp_file
                            for zp_file in zp_file_list
                            if anp_file in zp_file.lower()
                    ),
                    None,
            ):
                data = pd.read_csv(
                    BytesIO(zp.read(zp_file)), sep=None, engine="python"
                )
                for col in cols:
                    if col not in data.columns:
                        raise RuntimeError(
                            f"{col} is a mandatory column in the "
                            f"ANP table {anp_file}."
                        )

                setattr(self, anp_file, data)
            else:
                raise RuntimeError(
                    f"The ANP table {anp_file} was not found "
                    f"in the zip file {path.as_posix()}."
                )

    def _substitution_from_xlsx(self, path: Path) -> None:
        self.substitution = pd.read_excel(
            path,
            sheet_name="by ICAO code",
            usecols=[
                "ICAO_CODE",
                "AIRCRAFT_VARIANT",
                "ANP_PROXY",
                "DELTA_DEP_dB",
                "DELTA_APP_dB",
            ],
            keep_default_na=False,
        )
