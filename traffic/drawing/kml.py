from collections import UserDict
from contextlib import contextmanager
from typing import Dict, Optional, Union

from fastkml import LineStyle, PolyStyle, kml

_colormap = {
    'blue': '#1928f0',
    'red': '#ff3900',
    'green': '#19ff00',
    'yellow': '#effe00',
    'magenta': '#f300b0',
    'cyan': '#02d8ec',
    'orange': '#ffc000',
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

_stylemap = StyleMap()


def toStyle(color: str, alpha: float=0.5) -> kml.StyleUrl:
    # saturate alpha
    alpha = max(0, alpha)
    alpha = min(1, alpha)
    color = _colormap.get(color, color).upper()
    if len(color) != 7:
        raise ValueError("Invalid color")
    color = color[1:]
    # KML colors are f***ed...
    key = hex(int(alpha*255))[2:] + color[4:6] + color[2:4] + color[:2]
    return _stylemap[key]


@contextmanager
def export(filename: str):
    global current_document
    kml_tree = kml.KML()
    current_document = kml.Document()
    yield current_document
    kml_tree.append(current_document)
    with open(filename, 'w', encoding="utf8") as kml_file:
        kml_file.write(kml_tree.to_string(prettyprint=True))
    _stylemap.clear()
    current_document = None
