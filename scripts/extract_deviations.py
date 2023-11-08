from __future__ import annotations
from traffic.data import (  # noqa: F401
    aixm_airspaces,
    aixm_navaids,
    aixm_airways,
)
from traffic.core import Traffic, Flight, FlightPlan
import pandas as pd

from pathlib import Path
import datetime

from typing import Any, Dict, cast  # noqa: F401
from stats_devs_pack.functions_heuristic import predict_fp
import os

from traffic.core.mixins import DataFrameMixin
import multiprocessing as mp
from typing import Tuple, List

extent = "LFBBBDX"
prefix_sector = "LFBB"
margin_fl = 50  # margin for flight level
altitude_min = 20000
angle_precision = 2
forward_time = 20
min_distance = 200


class Metadata(DataFrameMixin):
    def __getitem__(self, key: str) -> None | FlightPlan:
        df = self.data.query(f'flight_id == "{key}"')
        if df.shape[0] == 0:
            return None
        return FlightPlan(df.iloc[0]["route"])


# metadata = cycle airac complet
metadata = pd.read_parquet("A2207_old.parquet")

metadata_simple = Metadata(
    metadata.groupby("flight_id", as_index=False)
    .last()
    .eval("icao24 = icao24.str.lower()")
)

# SEPARATION

print("avant chargement t2")
t2 = Traffic.from_file(
    "t2_0722_noonground2.parquet"
)  # A CHANGER POUR LE NOUVEAU
assert t2 is not None

# SEPARATION
print("post chargement t2")


def dist_lat_min(f1: Flight, f2: Flight) -> Any:
    try:
        if f1 & f2 is None:  # no overlap
            print(f"no overlap with {f2.flight_id}")
            return None
        return cast(pd.DataFrame, f1.distance(f2))["lateral"].min()
    except TypeError as e:
        print(
            f"exception in dist_lat_min for flights {f1.flight_id} and {f2.flight_id}"
        )
        return None


# SEPARATION


# we eliminate an abnormal flight causing problems
probleme = t2["AA39472649"]
assert probleme is not None
t2 = t2 - probleme
assert t2 is not None

nbworkers = 60
nt2 = len(t2)
nbsubsets = nt2 // nbworkers

# POUR LE BUG
subsetst2 = [
    t2[nbsubsets * i : nbsubsets * (i + 1)]
    if i < nbworkers - 1
    else t2[nbsubsets * i :]
    for i in range(nbworkers)
]

# POUR LE BUG
couples_datas = [
    (sub, f"test_extract_deviations/stats_para_{i}.parquet")
    for i, sub in enumerate(subsetst2)
]


def traitement(info: Tuple[Traffic, str]) -> None:
    subset, sortie = info
    list_dicts = []
    ids_error = []
    for flight in subset:
        id = flight.flight_id
        assert isinstance(id, str)
        try:
            # fp = metadata_simple[id]
            for flight_trou in flight - flight.aligned_on_navpoint(
                metadata_simple[id],
                angle_precision=angle_precision,
                min_distance=min_distance,
            ):
                temp_dict = flight_trou.summary(
                    ["flight_id", "start", "stop", "duration"]
                )
                if (
                    flight_trou is not None
                    and flight_trou.duration > pd.Timedelta("120s")
                    and flight_trou.altitude_max - flight_trou.altitude_min
                    < margin_fl
                    and flight_trou.start > flight.start
                    and flight_trou.stop < flight.stop
                ):
                    flight = flight.resample("1s")
                    flight_trou = flight_trou.resample("1s")

                    flmin = flight_trou.altitude_min - margin_fl
                    flmax = flight_trou.altitude_max + margin_fl
                    # START CREATING NEIGHBOURS
                    stop_voisins = min(
                        flight_trou.start + pd.Timedelta(minutes=forward_time),
                        flight.stop,
                    )
                    flight_interest = flight.between(
                        flight_trou.start, stop_voisins
                    )
                    assert flight_interest is not None
                    # find potential off-limits portion(s) in terms of altitude
                    offlimits = flight_interest.query(
                        f"altitude>{flmax} or altitude<{flmin}"
                    )
                    # if there is at least one off-limits portion, we cut
                    if offlimits is not None:
                        istop = offlimits.data.index[0]
                        flight_interest.data = flight_interest.data.loc[:istop]
                        stop_voisins = flight_interest.stop

                    voisins = (
                        (t2 - flight)
                        .between(
                            start=flight_trou.start,
                            stop=stop_voisins,
                            strict=False,
                        )
                        .iterate_lazy()
                        .query(f"{flmin} <= altitude <= {flmax}")
                        .feature_gt("duration", datetime.timedelta(seconds=2))
                        # .resample("1s")
                        .eval()
                    )
                    # STOP CREATING NEIGHBOURS
                    pred_possible = flight.before(flight_trou.start) is not None

                    if voisins is None and not pred_possible:
                        temp_dict = {
                            **temp_dict,
                            **dict(
                                nb_voisins=0,
                                min_f_dist=None,
                                min_f_id=None,
                                min_p_dist=None,
                                min_p_id=None,
                                max_dev_angle=None,
                                q90_dev_angle=None,
                                deviation_area=None,
                                min_f_dev=None,  # deviation time for closest
                            ),
                        }
                        continue

                    if pred_possible:
                        # compute prediction
                        pred = (
                            flight.before(flight_trou.start)  # type: ignore
                            .forward(minutes=forward_time)
                            .resample("1s")
                        )
                        pred_fp = predict_fp(
                            flight,
                            metadata_simple[id],
                            flight_trou.start,
                            flight_trou.stop,
                            minutes=forward_time,
                            min_distance=min_distance,
                        )

                        # angle computation (max and 90% quantile)
                        angle_flown = flight_trou.data.set_index("timestamp")[
                            "track"
                        ]
                        angle_pred = pred.data.set_index("timestamp")["track"]
                        d = pd.DataFrame((angle_flown - angle_pred) % 360)
                        d.loc[d["track"] < -180, "track"] = d["track"] + 360
                        d.loc[d["track"] >= 180, "track"] = d["track"] - 360
                        temp_dict["max_dev_angle"] = d.track.abs().max()
                        temp_dict["q90_dev_angle"] = (
                            d.track.abs().quantile([0.9]).iloc[0]
                        )

                        # total distance (sum of lateral pred-flown)
                        temp_dict["deviation_area"] = flight_trou.distance(
                            pred
                        ).lateral.sum()

                        # zero neighbors
                        temp_dict["nb_voisins"] = 0

                    if voisins is not None:
                        temp_dict["nb_voisins"] = len(voisins)
                        # distance to closest neighbor + flight_id + timestamp

                        (min_f, idmin_f) = min(
                            (dist_lat_min(flight_interest, f), f.flight_id)
                            for f in voisins
                        )
                        temp_dict["neighbour_id"] = idmin_f
                        temp_dict["min_f_dist"] = min_f
                        temp_dict["min_f_id"] = idmin_f
                        df_dist = flight_interest.distance(voisins[idmin_f])
                        temp_dict["min_f_time"] = df_dist.loc[
                            df_dist.lateral == df_dist.lateral.min()
                        ].timestamp.iloc[0]

                        # length of neighbor deviation
                        temp_dict["min_f_dev"] = pd.Timedelta(
                            seconds=sum(
                                f.duration.total_seconds()
                                for f in (
                                    voisins[idmin_f]
                                    - voisins[idmin_f].aligned_on_navpoint(
                                        metadata_simple[idmin_f],
                                        angle_precision,
                                        min_distance=min_distance,
                                    )
                                )
                            )
                        )
                        if pred_possible:
                            # we use the same neighbor to compute predicted distance
                            df_dist_forward = pred.distance(voisins[idmin_f])
                            temp_dict["min_p_id"] = idmin_f
                            temp_dict[
                                "min_p_dist"
                            ] = df_dist_forward.lateral.min()
                            temp_dict["min_p_time"] = df_dist_forward.loc[
                                df_dist_forward.lateral
                                == df_dist_forward.lateral.min()
                            ].timestamp.iloc[0]

                            df_dist_fp = pred_fp.distance(voisins[idmin_f])
                            temp_dict["min_fp_id"] = idmin_f
                            temp_dict["min_fp_dist"] = df_dist_fp.lateral.min()
                            temp_dict["min_fp_time"] = df_dist_fp.loc[
                                df_dist_fp.lateral == df_dist_fp.lateral.min()
                            ].timestamp.iloc[0]

                    list_dicts.append(temp_dict)

        except AssertionError:
            ids_error.append(id)
        except TypeError as e:
            print(f"TypeError in main for flight {id}")
        except AttributeError as e:
            print(f"AttributeError in main for flight {id}")

    df = pd.DataFrame(list_dicts)
    df.to_parquet(sortie, index=False)


def do_parallel(
    f: function,
    datas: List[Tuple[Traffic, str]],
    nworkers: int = mp.cpu_count() // 2,
) -> None:
    with mp.Pool(nworkers) as pool:
        l = pool.map(f, datas, chunksize=1)


do_parallel(traitement, couples_datas, nbworkers)
