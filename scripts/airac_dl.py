import pandas as pd
from atmlab.mongo import Mongo
from tqdm.autonotebook import tqdm

with Mongo("ops").client() as client:
    query_regex = client.flightdata.A2207.find(
        {"flightData.ctfmAirspaceProfile.airspaceId": {"$regex": "^LFBB.*"}},
        {
            "flightData.actualOffBlockTime": True,
            "flightData.actualTimeOfArrival": True,
            "flightData.aircraftAddress": True,
            "flightData.aircraftType": True,
            "flightData.flightId.id": True,
            "flightData.flightId.keys.aerodromeOfDeparture": True,
            "flightData.flightId.keys.aerodromeOfDestination": True,
            "flightData.flightId.keys.aircraftId": True,
            "flightData.flightId.keys.estimatedOffBlockTime": True,
            "flightData.icaoRoute": True,
            "flightData.flightState": True,
        },
    )
    results = pd.json_normalize(tqdm(query_regex))

convert_types = {"callsign": str, "icao24": str, "flight_id": str}
results = (
    results.rename(
        columns={
            "flightData.actualOffBlockTime": "start",
            "flightData.actualTimeOfArrival": "stop",
            "flightData.aircraftAddress": "icao24",
            "flightData.aircraftType": "typecode",
            "flightData.flightId.id": "flight_id",
            "flightData.flightId.keys.aerodromeOfDeparture": "origin",
            "flightData.flightId.keys.aerodromeOfDestination": "destination",
            "flightData.flightId.keys.aircraftId": "callsign",
            "flightData.flightId.keys.estimatedOffBlockTime": "EOBT",
            "flightData.icaoRoute": "route",
            "flightData.flightState": "state",
        }
    )
    .astype(convert_types)
    .convert_dtypes()
    .assign(
        start=lambda df: pd.to_datetime(df["start"], utc=True),
        stop=lambda df: pd.to_datetime(df["stop"], utc=True),
        EOBT=lambda df: pd.to_datetime(df["EOBT"], utc=True),
    )
)

# df_euro.to_parquet("data_euro_22-07-13.parquet")

results.drop(columns=["_id"]).groupby(
    "flight_id", as_index=False
).last().to_parquet("A2207.parquet")
