from collections import UserDict
from contextlib import contextmanager
from typing import Optional

import numpy as np

from fastkml import LineStyle, PolyStyle, kml
from fastkml.geometry import Geometry
from shapely.geometry import LineString

from ..core import Flight, Airspace

_colormap = {
    "blue": "#1928f0",
    "red": "#ff3900",
    "green": "#19ff00",
    "yellow": "#effe00",
    "magenta": "#f300b0",
    "cyan": "#02d8ec",
    "orange": "#ffc000",
}

current_document: Optional[kml.Document] = None


class StyleMap(UserDict):
    def __missing__(self, color: str):
        style = kml.Style(id=color)
        if current_document is not None:
            current_document.append_style(style)
        style.append_style(LineStyle(color=color, width=2))
        style.append_style(PolyStyle(color=color))
        styleUrl = kml.StyleUrl(url=color)
        return styleUrl


def toStyle(color: str, alpha: float = 0.5) -> kml.StyleUrl:
    # saturate alpha
    alpha = max(0, alpha)
    alpha = min(1, alpha)
    color = _colormap.get(color, color).upper()
    if len(color) != 7:
        raise ValueError("Invalid color")
    color = color[1:]
    # KML colors are f***ed...
    key = hex(int(alpha * 255))[2:] + color[4:6] + color[2:4] + color[:2]
    return _stylemap[key]


@contextmanager
def export(filename: str):
    global current_document
    kml_tree = kml.KML()
    current_document = kml.Document()
    yield current_document
    kml_tree.append(current_document)
    with open(filename, "w", encoding="utf8") as kml_file:
        kml_file.write(kml_tree.to_string(prettyprint=True))
    _stylemap.clear()
    current_document = None


def _flight_export_kml(
    flight: Flight,
    styleUrl: Optional[kml.StyleUrl] = None,
    color: Optional[str] = None,
    alpha: float = .5,
    **kwargs,
) -> kml.Placemark:
    if color is not None:
        # the style will be set only if the kml.export context is open
        styleUrl = toStyle(color)
    params = {
        "name": flight.callsign,
        "description": flight._info_html(),
        "styleUrl": styleUrl,
    }
    for key, value in kwargs.items():
        params[key] = value
    placemark = kml.Placemark(**params)
    placemark.visibility = 1
    # Convert to meters
    coords = np.stack(flight.coords)
    coords[:, 2] *= 0.3048
    placemark.geometry = Geometry(
        geometry=LineString(coords),
        extrude=True,
        altitude_mode="relativeToGround",
    )
    return placemark


def _airspace_export_kml(
    sector: Airspace,
    styleUrl: Optional[kml.StyleUrl] = None,
    color: Optional[str] = None,
    alpha: float = .5,
) -> kml.Placemark:
    if color is not None:
        # the style will be set only if the kml.export context is open
        styleUrl = toStyle(color)
    folder = kml.Folder(name=sector.name, description=sector.type)
    for extr_p in sector:
        for elt in sector.decompose(extr_p):
            placemark = kml.Placemark(styleUrl=styleUrl)
            placemark.geometry = kml.Geometry(
                geometry=elt, altitude_mode="relativeToGround"
            )
            folder.append(placemark)
    return folder


setattr(Flight, "export_kml", _flight_export_kml)
setattr(Airspace, "export_kml", _airspace_export_kml)
_stylemap = StyleMap()
