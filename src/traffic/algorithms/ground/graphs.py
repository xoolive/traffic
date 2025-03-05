import json
import logging
from collections import defaultdict
from itertools import pairwise
from typing import Any, Iterator, TypeVar

import geopandas as gpd
import networkx as nx
from cartes.crs import EuroPP, Projection  # type: ignore
from ipyleaflet import Marker, Polyline
from ipywidgets import HTML
from scipy.spatial import KDTree
from typing_extensions import Self

import numpy as np
import pandas as pd
from pyproj import Proj, Transformer
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import split, transform

from ...core import Flight
from ...core.structure import Airport

projection = EuroPP()

T = TypeVar("T", bound=BaseGeometry)
_log = logging.getLogger(__name__)


class AirportGraph:
    """Create a graph from an airport."""

    wgs84 = Proj("EPSG:4326")

    @classmethod
    def from_airport(cls, airport: Airport, projection: Projection) -> Self:
        graph = airport._openstreetmap().network_graph(
            "geometry",
            "aeroway",
            "parking_position",
            "ref",
            "name",
            query_str='aeroway == "taxiway" or aeroway == "runway" '
            'or aeroway == "parking_position"',
        )
        return cls(graph, projection).fix_airport_graph()

    def __init__(self, graph: nx.MultiGraph, projection: Projection) -> None:
        self.graph = graph
        self.projection = projection
        self.project = Transformer.from_proj(
            self.wgs84, self.projection, always_xy=True
        )
        self.to_wgs84 = Transformer.from_proj(
            self.projection, self.wgs84, always_xy=True
        )

        # this helps finding the closest node along the trajectory
        self.node_map = [node for node, _data in self.graph.nodes(data=True)]
        self.node_kdtree = KDTree(
            [
                self.project.transform(*data["pos"])
                for _node, data in self.graph.nodes(data=True)
            ]
        )

    @property
    def components(self) -> list[int]:
        """Confirm the number of connected components"""
        return list(len(p) for p in nx.connected_components(self.graph))

    def filter_connected_components(self) -> None:
        nodes = max(nx.connected_components(self.graph), key=len)
        self.node_map = [
            node for node, _data in self.graph.nodes(data=True) if node in nodes
        ]
        self.node_kdtree = KDTree(
            [
                self.project.transform(*data["pos"])
                for node, data in self.graph.nodes(data=True)
                if node in nodes
            ]
        )

    def merge_duplicate_nodes(self) -> Self:
        """Relabel the graph to deal with nodes with identical coordinates.

        This function helps rebuilding a topologically correct graph based on
        OpenStreetMap data.

        Among the issues with raw OpenStreetMap data, we are bothered by few
        nodes which have a different id but the same coordinates.

        """
        pos_to_nodes = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            pos = data.get("pos")
            if pos is not None:
                pos_to_nodes[pos].append(node)
        mapping = {}
        for pos, nodes in pos_to_nodes.items():
            if len(nodes) > 1:
                # Keep the first node and map the others to it
                representative = nodes[0]
                for node in nodes[1:]:
                    mapping[node] = representative

        graph: nx.MultiGraph = nx.relabel_nodes(self.graph, mapping, copy=False)

        # Relabel first and last fields according to the mapping
        for _, _, data in graph.edges(data=True):
            data["first"] = mapping.get(data["first"], data["first"])
            data["last"] = mapping.get(data["last"], data["last"])

        return AirportGraph(graph, self.projection)  # type: ignore

    def buffer_meter(
        self, shape: BaseGeometry, buffer: int = 18
    ) -> BaseGeometry:
        """Add a buffer around the geometric shape.

        Useful to simulate for the width of taxiways.
        The buffer parameter is in meters.
        """
        projected = transform(self.project.transform, shape)
        buffered = projected.buffer(buffer)
        return transform(self.to_wgs84.transform, buffered)

    def project_shape(self, shape: T) -> T:
        return transform(self.project.transform, shape)  # type: ignore

    def length(self, shape: BaseGeometry) -> float:
        return self.project_shape(shape).length  # type: ignore

    def split_line_based_on_point(
        self, line: LineString, splitter: BaseGeometry
    ) -> list[LineString]:
        """Split a LineString with a Point"""

        # point is on line, get the distance from the first point on line
        distance_on_line = line.project(splitter)
        coords = list(line.coords)
        # split the line at the point and create two new lines
        current_position = 0.0
        for i in range(len(coords) - 1):
            point1 = coords[i]
            point2 = coords[i + 1]
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            segment_length = (dx**2 + dy**2) ** 0.5
            current_position += segment_length
            if distance_on_line == current_position:
                # splitter is exactly on a vertex
                return [
                    LineString(coords[: i + 2]),
                    LineString(coords[i + 1 :]),
                ]
            elif distance_on_line < current_position:
                # splitter is between two vertices
                return [
                    LineString(coords[: i + 1] + [splitter.coords[0]]),
                    LineString([splitter.coords[0]] + coords[i + 1 :]),
                ]
        return [line]

    def snap_and_split(
        self,
        graph: nx.MultiGraph,
        u: int,
        v: int,
        k: int,
        tolerance: float,
    ) -> None:
        """Recursive companion function to split edges when an extra node is
        located on another edge, without being topologically connected.

        - u, v are the indices of the nodes
        - k is the key (because of the MultiGraph)
        - tolerance (in m) allows snapping points that do not fall exactly on a
            segment
        """
        _log.debug(f"enter snap_and_split for nodes {u=} and {v=} ")

        edge_data = graph.get_edge_data(u, v, k)
        u, v = edge_data["first"], edge_data["last"]
        line_wgs84 = edge_data["geometry"]
        line_proj = self.project_shape(line_wgs84)

        ((x, y),) = line_proj.centroid.coords
        _dist, idx = self.node_kdtree.query(
            (x, y), distance_upper_bound=line_proj.length / 2, k=30
        )
        candidate_idx = [
            self.node_map[i] for i in idx if i < len(self.node_map)
        ]
        min_list = [
            (
                line_proj.distance(self.project_shape(Point(*data["pos"]))),
                i,
                Point(*data["pos"]),
            )
            for i, data in graph.nodes(data=True)
            if i in candidate_idx and i != u and i != v
        ]
        if len(min_list) == 0:
            return
        min_dist, min_idx, min_point = min(min_list)

        _log.debug(f"{min_dist=}, {min_idx=}, {u=}, {v=}")

        if min_dist < tolerance:
            projected_point = line_wgs84.interpolate(
                line_wgs84.project(min_point)
            )
            if line_wgs84.distance(projected_point) > 0:
                splits = self.split_line_based_on_point(
                    line_wgs84, projected_point
                )
            else:
                splits = split(line_wgs84, projected_point).geoms

            splits_iter = iter(splits)
            left = next(splits_iter)
            right = next(splits_iter, None)
            if right is None:
                # TODO prepare a warning here
                return

            _log.debug(f"split ({u=}, {v=}) into ({u=}, {min_idx=}, {v=})")

            graph.remove_edge(u, v, key=k)
            left_key = graph.add_edge(
                u,
                min_idx,
                **{**edge_data, "geometry": left, "first": u, "last": min_idx},
            )
            right_key = graph.add_edge(
                min_idx,
                v,
                **{**edge_data, "geometry": right, "first": min_idx, "last": v},
            )

            self.snap_and_split(graph, u, min_idx, left_key, tolerance)
            self.snap_and_split(graph, min_idx, v, right_key, tolerance)

    def fix_airport_graph(self) -> Self:
        """Builds an airport graph based on available information.

        - merge duplicate nodes
        - for each edge, detect every node that is geographically on the segment
          (linestring) and split the edge into two.

        The resulting graph should be (at least very close to) a single
        connected component graph.
        """
        airport_graph = self.merge_duplicate_nodes()
        for u, v, k in list(airport_graph.graph.edges):
            self.snap_and_split(airport_graph.graph, u, v, k, 18)
        return airport_graph

    def intersection_with_flight(self, flight: Flight) -> pd.DataFrame:
        """Compute the intersection of a flight trajectory with all edges of the
        airport graph."""

        df = pd.DataFrame.from_records(
            data for u, v, data in self.graph.edges(data=True)
        )
        gdf = gpd.GeoDataFrame(df).set_crs(epsg=4326).to_crs(projection)
        buffered_gdf = gdf.to_crs(projection).buffer(18).to_crs(epsg=4326)
        intersected_pieces = df.loc[buffered_gdf.intersects(flight.shape)]
        segment = gdf.loc[buffered_gdf.intersects(flight.shape)].geometry
        trajectory_on_segment = buffered_gdf.loc[
            buffered_gdf.intersects(flight.shape)
        ].intersection(flight.shape)

        return (
            intersected_pieces.assign(
                distance_difference=segment.to_crs(projection).length
                - trajectory_on_segment.to_crs(projection).length
            )
            .assign(
                score=lambda df: np.where(
                    df["distance_difference"] > 0,
                    df["distance_difference"],
                    0,
                )
            )
            .query("score.notnull()")
        )

    def map_flight(self, g: Flight) -> Iterator[dict[str, Any]]:
        """Get the most probable path along edges with a pathfinding algorithm.

        Returns segment one after the other.
        """
        intersection = self.intersection_with_flight(g)
        copy_graph = nx.Graph(self.graph)
        for u, v, data in copy_graph.edges(data=True):
            data["distance"] = self.length(data["geometry"])
        for u, v, data in zip(
            intersection["first"],
            intersection["last"],
            intersection["score"],
        ):
            copy_graph.get_edge_data(u, v)["distance"] = data

        # TODO improve x0 and x1 detection based on intersection
        # TODO with parking positions
        x0 = g.at_ratio(0)
        x1 = g.at_ratio(1)
        assert x0 is not None
        assert x1 is not None
        ((x, y),) = self.project_shape(Point(x0.longitude, x0.latitude)).coords
        _, node0 = self.node_kdtree.query((x, y))
        ((x, y),) = self.project_shape(Point(x1.longitude, x1.latitude)).coords
        _, node1 = self.node_kdtree.query((x, y))

        path = nx.shortest_path(
            copy_graph,
            self.node_map[node0],
            self.node_map[node1],
            weight="distance",
        )

        for u, v in pairwise(path):
            yield copy_graph.get_edge_data(u, v)

    def make_parts(self, flight: Flight) -> pd.DataFrame:
        """Make a table with statistics of the trajectory along the highlighted
        segments.
        """
        cumul = []
        flight_forward: None | Flight = flight

        for edge_data in self.map_flight(flight):
            width = 8 if edge_data["aeroway"] == "parking_position" else 18
            shape = self.buffer_meter(edge_data["geometry"], width)
            if edge_data["aeroway"] == "taxiway" and flight_forward is not None:
                # limit the noise in taxiways
                # but keep runways and parking positions
                intersection = flight_forward.clip(shape, strict=False)
            else:
                intersection = flight.clip(shape, strict=False)

            edge = edge_data.copy()
            edge["icao24"] = flight.icao24
            edge["callsign"] = flight.callsign
            edge["flight_id"] = flight.flight_id

            if intersection is not None:
                edge["start"] = intersection.start
                edge["stop"] = intersection.stop

                if flight_forward is not None:
                    flight_forward = flight_forward.after(intersection.stop)

            cumul.append(edge)

        return pd.DataFrame.from_records(cumul)

    def marker_node(self, id: int) -> Marker:
        """Visualisation function to include nodes on a Leaflet widget."""
        for node, data in self.graph.nodes(data=True):
            if node == id:
                lat, lon = data["pos"]
                marker = Marker(location=(lon, lat))
                marker.popup = HTML()
                marker.popup.value = f"Node {node}"
                return marker

    def marker_edge(self, data: dict[str, Any], **kwargs: Any) -> Polyline:
        """Visualisation function to include edges on a Leaflet widget."""
        coords = list((lat, lon) for (lon, lat, *_) in data["geometry"].coords)
        kwargs = {**dict(fill_opacity=0, weight=3), **kwargs}
        polyline = Polyline(locations=coords, **kwargs)
        polyline.popup = HTML()
        copy_data = data.copy()
        del copy_data["geometry"]
        polyline.popup.value = json.dumps(copy_data)
        return polyline
