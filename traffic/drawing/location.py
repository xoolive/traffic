import logging
from typing import Dict, List

import requests
from shapely.geometry import shape

from ..core.mixins import ShapelyMixin


def nominatim_request(query: str, **kwargs):

    params = {
        "format": "json",
        "limit": 1,
        "dedupe": 0,
        "polygon_geojson": 1,
        "q": query,
    }
    url = "https://nominatim.openstreetmap.org/search"
    response = requests.post(url, timeout=30, params=params, **kwargs)
    try:
        json = response.json()
    except Exception as e:
        logging.exception(e)  # type: ignore

    return json


class Nominatim(ShapelyMixin):

    def __init__(self, name: str, **kwargs) -> None:

        results: List[Dict] = nominatim_request(name, **kwargs)

        if len(results) == 0:
            raise ValueError(f"No '{name}' found on OpenStreetMap")

        self.json = results[0]
        self.shape = shape(self.json["geojson"])


def location(query: str, **kwargs):
    return Nominatim(query, **kwargs)
