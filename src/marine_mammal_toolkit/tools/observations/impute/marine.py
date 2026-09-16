"""Marine routing backed by the canonical seascape water graph."""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from pathlib import Path

import h3
import numpy as np

from seascape.spatial_support.water_network import WaterGraph
from marine_mammal_toolkit.tools._core.seascape import load_water_graph


class MarineDistanceLookup:
    """Resolve bounded marine routes through checksum-verified passable edges.

    Coordinates are mapped to the configured canonical H3 graph. Terminal water
    cells use only the connector recorded by the spatial-support product; unknown
    or disconnected cells remain unavailable rather than falling through land.
    """

    def __init__(
        self,
        config_path: str | Path,
        *,
        resolution: int,
        required_radius_km: float,
    ):
        self.path = Path(config_path).expanduser().resolve()
        self.resolution = int(resolution)
        self.max_distance_km = float(required_radius_km)
        self.graph = load_water_graph(self.resolution, self.path)
        self.known_cells = set(self.graph.support["H3_INDEX"].astype(str))
        self._support = self.graph.support.set_index("H3_INDEX", drop=False)
        self._route_cache: OrderedDict[str, dict[int, float]] = OrderedDict()
        self._route_cache_limit = 512

    def is_known_cell(self, cell: str) -> bool:
        return str(cell) in self.known_cells

    def _graph_mapping(self, cell: str) -> tuple[int, float]:
        value = str(cell)
        direct = self.graph.cell_to_position.get(value)
        if direct is not None:
            return direct, 0.0
        if value not in self._support.index:
            return -1, float("nan")
        row = self._support.loc[value]
        target = row.get("CONNECTOR_TARGET_H3_INDEX")
        distance = row.get("CONNECTOR_DISTANCE_M")
        if target is None or distance is None:
            return -1, float("nan")
        try:
            if not math.isfinite(float(distance)):
                return -1, float("nan")
        except (TypeError, ValueError):
            return -1, float("nan")
        position = self.graph.cell_to_position.get(str(target))
        return (
            (position, float(distance) / 1000.0)
            if position is not None
            else (-1, float("nan"))
        )

    def snap_distance_km(self, cell: str) -> float:
        _position, distance = self._graph_mapping(str(cell))
        return distance

    def cells_for_coordinates(
        self, latitude: np.ndarray, longitude: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            [
                h3.latlng_to_cell(float(lat), float(lon), self.resolution)
                for lat, lon in zip(latitude, longitude, strict=True)
            ],
            dtype=object,
        )

    def _distances_from(self, target_cell: str) -> dict[int, float]:
        cached = self._route_cache.get(target_cell)
        if cached is not None:
            self._route_cache.move_to_end(target_cell)
            return cached
        position, initial_km = self._graph_mapping(target_cell)
        distances: dict[int, float] = {}
        if position < 0 or not math.isfinite(initial_km):
            self._route_cache[target_cell] = distances
            if len(self._route_cache) > self._route_cache_limit:
                self._route_cache.popitem(last=False)
            return distances
        distances[position] = initial_km
        queue: list[tuple[float, int]] = [(initial_km, position)]
        while queue:
            current, node = heapq.heappop(queue)
            if current > distances[node] + 1e-12:
                continue
            if current > self.max_distance_km:
                continue
            neighbors, weights_m = self.graph.neighbors_of(node)
            for neighbor, weight_m in zip(neighbors, weights_m, strict=True):
                neighbor_position = int(neighbor)
                candidate = current + float(weight_m) / 1000.0
                if candidate > self.max_distance_km or candidate >= distances.get(
                    neighbor_position, float("inf")
                ):
                    continue
                distances[neighbor_position] = candidate
                heapq.heappush(queue, (candidate, neighbor_position))
        self._route_cache[target_cell] = distances
        if len(self._route_cache) > self._route_cache_limit:
            self._route_cache.popitem(last=False)
        return distances

    def resolve(
        self,
        target_cell: str,
        anchor_cells: np.ndarray,
        haversine_km: np.ndarray,
        *,
        fallback_to_haversine: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve routes without treating graph disconnection as zero or straight line."""

        target = str(target_cell)
        resolved = np.full(len(anchor_cells), np.nan, dtype=float)
        fallback = np.zeros(len(anchor_cells), dtype=bool)
        target_known = self.is_known_cell(target)
        graph_distances = self._distances_from(target)
        for index, anchor_object in enumerate(anchor_cells):
            anchor = str(anchor_object)
            if anchor == target and target_known:
                resolved[index] = 0.0
                continue
            position, connector_km = self._graph_mapping(anchor)
            if position >= 0 and math.isfinite(connector_km):
                route = graph_distances.get(position, float("inf")) + connector_km
                if route <= self.max_distance_km:
                    resolved[index] = route
                    continue
            anchor_known = self.is_known_cell(anchor)
            if fallback_to_haversine and not (target_known and anchor_known):
                resolved[index] = float(haversine_km[index])
                fallback[index] = True
        keep = np.isfinite(resolved)
        return resolved, keep, fallback
