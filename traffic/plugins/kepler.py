import logging
from collections.abc import Generator
from typing import Any, Dict, Iterator, List, Protocol, Union, cast

from keplergl import KeplerGl

import pandas as pd
from shapely.geometry import mapping
from shapely.wkt import dumps

from ..core import Airspace, Flight, Traffic


def flight_kepler(flight: "Flight") -> Dict[str, Any]:
    return dict(
        geometry=mapping(flight.shape),
        properties={
            "icao24": flight.icao24,
            "callsign": flight.callsign,
            "registration": flight.registration,
            "start": f"{flight.start:%Y-%m-%d %H:%M:%S}",
            "stop": f"{flight.stop:%Y-%m-%d %H:%M:%S}",
        },
        type="Feature",
    )


def airspace_kepler(airspace: "Airspace") -> Dict[str, Any]:
    return dict(
        geometry=mapping(airspace.shape),
        properties={
            "name": airspace.name,
            "designator": airspace.designator,
            "type": airspace.type,
        },
        type="Feature",
    )


def traffic_kepler(traffic: "Traffic") -> pd.DataFrame:
    if traffic.flight_ids is None:
        logging.warning("assign_id() has been appended for you")
        traffic = cast("Traffic", traffic.assign_id().eval())
    return pd.DataFrame.from_records(
        [
            {
                "id": flight.flight_id
                if flight.flight_id is not None
                else flight.aircraft,
                "wkt_string": dumps(flight.shape),
                "icao24": flight.icao24,
                "callsign": flight.callsign,
                "registration": flight.registration,
                "start": f"{flight.start:%Y-%m-%d %H:%M:%S}",
                "stop": f"{flight.stop:%Y-%m-%d %H:%M:%S}",
            }
            for flight in traffic
        ]
    )


_old_add_data = KeplerGl.add_data


class HasData(Protocol):
    data: pd.DataFrame


Mappable = Union[Airspace, Flight, Traffic, HasData]


def map_add_data(
    _map: KeplerGl,
    data: Union[Mappable, List[Mappable], Iterator[Mappable], HasData],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if any(isinstance(data, c) for c in (Airspace, Flight, Traffic)):
        layer = data.kepler()  # type: ignore
        return _old_add_data(_map, layer, *args, **kwargs)

    if any(isinstance(data, c) for c in (list, Generator)):
        layer = dict(
            type="FeatureCollection",
            features=[elt.kepler() for elt in data],  # type: ignore
        )
        return _old_add_data(_map, layer, *args, **kwargs)

    # convenient for airports, navaids, etc.
    if hasattr(data, "data"):
        data = data.data  # type: ignore

    return _old_add_data(_map, data, *args, **kwargs)


def _onload() -> None:
    setattr(Airspace, "kepler", airspace_kepler)
    setattr(Flight, "kepler", flight_kepler)
    setattr(Traffic, "kepler", traffic_kepler)
    setattr(KeplerGl, "add_data", map_add_data)
