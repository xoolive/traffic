from traffic.data import airports

airports.query(
    'icao.str.match("[A-Z]{4}") and iata.notnull() '
    'and type != "closed" and type != "small_airport" '
    'and type != "heliport" and type != "seaplane_base" '
    'and icao != "ZHZH"'  # misleading identifier for this Canadian airport
).sort_values(  # type: ignore
    "icao"
).to_csv(
    "airports.csv", index=False
)
