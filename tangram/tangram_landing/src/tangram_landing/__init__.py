import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import tangram_core
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from traffic.algorithms.navigation.landing import LandingAnyAttempt
from traffic.core import Flight
from traffic.data import airports

log = logging.getLogger(__name__)


router = APIRouter(
    prefix="/align",
    tags=["align"],
    responses={404: {"description": "Not found"}},
)


@dataclass(frozen=True)
class Payload:
    aircraft: list[Any]


airport_subset = airports.query('type in ["large_airport", "medium_airport"]')
method = LandingAnyAttempt(dataset=airport_subset)


@router.post("/airport")
def align_airport(payload: Payload) -> JSONResponse:
    """Align airports using tangram_align algorithms."""
    log.info("Aligning airports for aircraft")
    flight = Flight(
        pd.DataFrame.from_records(payload.aircraft).assign(
            timestamp=lambda df: pd.to_datetime(df.timestamp, unit="s", utc=True)
        )
    )
    landing = flight.resample("1s").landing(method=method).final()
    if landing is None:
        return JSONResponse(content={"status": "not found"})
    return JSONResponse(
        content={
            "status": "found",
            "airport": landing.airport_max,
            "runway": landing.ILS_max,
            "latlon": next(
                runway
                for runway in airports[landing.airport_max].runways.list
                if runway.name == landing.ILS_max
            ).latlon,
        },
    )


plugin = tangram_core.Plugin(
    frontend_path="dist-frontend",
    routers=[router],
)
