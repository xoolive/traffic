from dataclasses import dataclass
from typing import Any, Final, Literal

import pandas as pd
import tangram_core
from fastapi import APIRouter, HTTPException
from traffic.algorithms.navigation.landing import LandingAnyAttempt
from traffic.core import Flight
from traffic.data import airports
from typing_extensions import TypedDict

router = APIRouter(
    prefix="/align",
    tags=["align"],
    responses={404: {"description": "Not found"}},
)


@dataclass(frozen=True)
class Payload:
    aircraft: list[dict[str, Any]]


AIRPORT_SUBSET = airports.query('type in ["large_airport", "medium_airport"]')
LANDING_METHOD = LandingAnyAttempt(dataset=AIRPORT_SUBSET)

FLIGHT_GAP: Final = "30 min"
ANALYSIS_WINDOW: Final = "6h"
RESAMPLE_RULE: Final = "1s"
MAX_RESAMPLED_POINTS: Final = 25_000


def _prepare_flight(records: list[dict[str, Any]]) -> Flight:
    frame = (
        pd.DataFrame.from_records(records)
        .assign(
            timestamp=lambda data: pd.to_datetime(
                data.timestamp,
                unit="s",
                utc=True,
            )
        )
        .sort_values("timestamp")
    )
    # jet1090 returns all history for an icao24 so isolate the latest flight
    # before resampling, otherwise long gaps are interpolated at 1Hz
    # TODO query the bounded interval directly from redis instead of round
    # tripping the trajectory through the frontend.
    segment = list(Flight(frame).split(FLIGHT_GAP))[-1].last(ANALYSIS_WINDOW)

    if (
        estimated_points := int(
            (segment.stop - segment.start) / pd.Timedelta(RESAMPLE_RULE)
        )
        + 1
    ) > MAX_RESAMPLED_POINTS:
        raise HTTPException(
            status_code=422,
            detail=(
                "trajectory interval is too large to resample safely: "
                f"{estimated_points} points"
            ),
        )

    return segment


class AlignmentNotFoundResponse(TypedDict):
    status: Literal["not found"]


class AlignmentFoundResponse(TypedDict):
    status: Literal["found"]
    airport: str
    runway: str
    latlon: list[float]


@router.post("/airport")
def align_airport(
    payload: Payload,
) -> AlignmentFoundResponse | AlignmentNotFoundResponse:
    """Align the latest contiguous flight interval with its landing runway."""
    flight = _prepare_flight(payload.aircraft)

    if (
        landing := flight.resample(RESAMPLE_RULE).landing(method=LANDING_METHOD).final()
    ) is None:
        return {"status": "not found"}

    airport = airports[landing.airport_max]
    assert airport is not None and airport.runways is not None
    runway = next(
        candidate
        for candidate in airport.runways.list
        if candidate.name == landing.ILS_max
    )
    latitude, longitude = runway.latlon

    return {
        "status": "found",
        "airport": landing.airport_max,
        "runway": landing.ILS_max,
        "latlon": [float(latitude), float(longitude)],
    }


plugin = tangram_core.Plugin(
    frontend_path="dist-frontend",
    routers=[router],
)
