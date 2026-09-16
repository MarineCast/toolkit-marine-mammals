"""Water-constrained probability surfaces derived from a fitted imputer."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import h3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, laplacian
from scipy.sparse.linalg import expm_multiply
from sklearn.neighbors import BallTree

from seascape.spatial_support.water_network import WaterGraph
from seascape.spatial_support.water_network import load_water_graph

from marine_mammal_toolkit.cetaceans.killer_whales.observations.features import (
    EARTH_RADIUS_KM,
)
from marine_mammal_toolkit.tools.observations.impute.visualization import MAP_COLORS

if TYPE_CHECKING:
    from marine_mammal_toolkit.tools.observations.impute.model import (
        SelectiveDateContextImputer,
    )


SUPPORTED_SURFACE_REGIMES = {
    "SAME_DAY",
    "SAME_DAY_CONFLICT",
    "LAGGED",
    "LAGGED_CONFLICT",
    "WEAK_LOCAL",
}


def _canonical_water_cells(
    graph: WaterGraph,
    context: pd.DataFrame,
    *,
    parent_resolution: int,
    surface_resolution: int,
    radius_km: float,
    bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    cells = graph.cells.astype(str)
    centers = np.asarray([h3.cell_to_latlng(cell) for cell in cells], dtype=float)
    min_lat, max_lat, min_lon, max_lon = bounds
    in_bounds = (
        (centers[:, 0] >= min_lat)
        & (centers[:, 0] <= max_lat)
        & (centers[:, 1] >= min_lon)
        & (centers[:, 1] <= max_lon)
    )
    cells = cells[in_bounds]
    centers = centers[in_bounds]
    tree = BallTree(
        np.deg2rad(context[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)),
        metric="haversine",
    )
    near_context = (
        tree.query_radius(
            np.deg2rad(centers), r=radius_km / EARTH_RADIUS_KM, count_only=True
        )
        > 0
    )
    cells = cells[near_context]
    centers = centers[near_context]
    return pd.DataFrame(
        {
            "H3_R8": cells,
            "PARENT_H3": [h3.cell_to_parent(cell, parent_resolution) for cell in cells],
            "LATITUDE": centers[:, 0],
            "LONGITUDE": centers[:, 1],
        }
    )


def _water_graph(graph: WaterGraph, cells: list[str]) -> tuple[csr_matrix, csr_matrix]:
    positions = {cell: idx for idx, cell in enumerate(cells)}
    rows: list[int] = []
    cols: list[int] = []
    distances_km: list[float] = []
    for left_idx, cell in enumerate(cells):
        graph_position = graph.cell_to_position[cell]
        neighbors, weights_m = graph.neighbors_of(graph_position)
        for neighbor_position, weight_m in zip(neighbors, weights_m, strict=True):
            neighbor = str(graph.cells[int(neighbor_position)])
            right_idx = positions.get(neighbor)
            if right_idx is None or right_idx <= left_idx:
                continue
            rows.extend((left_idx, right_idx))
            cols.extend((right_idx, left_idx))
            weight_km = float(weight_m) / 1000.0
            distances_km.extend((weight_km, weight_km))
    values = np.ones(len(rows), dtype=np.float32)
    shape = (len(cells), len(cells))
    return (
        csr_matrix((values, (rows, cols)), shape=shape),
        csr_matrix((distances_km, (rows, cols)), shape=shape),
    )


def build_weekly_probability_surface(
    imputer: "SelectiveDateContextImputer",
    water_network_config_path: str,
    *,
    year: int = 2025,
    iso_week: int = 32,
    surface_resolution: int = 8,
    gaussian_sigma_km: float = 8.0,
    bounds: tuple[float, float, float, float] = (46.8, 51.2, -126.5, -121.5),
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Create a complete water-graph surface around temporally relevant evidence.

    The fitted model supplies locally supported seed probabilities. A normalized
    heat-kernel (Gaussian on the H3 water graph) fills intervening water cells.
    Cells disconnected from model-supported evidence remain absent.
    """

    imputer._check_fitted()
    lookup = imputer.feature_builder.marine_lookup
    if lookup is None:
        raise ValueError("A marine-distance lookup is required for a water surface")
    if surface_resolution <= lookup.resolution:
        raise ValueError("surface_resolution must exceed the marine lookup resolution")
    if gaussian_sigma_km <= 0:
        raise ValueError("gaussian_sigma_km must be positive")

    week_start = pd.Timestamp(date.fromisocalendar(year, iso_week, 1))
    week_end = pd.Timestamp(date.fromisocalendar(year, iso_week, 7))
    surface_date = week_start + pd.Timedelta(days=3)
    anchors = imputer.anchors_.copy()  # type: ignore[union-attr]
    anchors["SIGHTING_DATE"] = pd.to_datetime(
        anchors["SIGHTING_DATE"], format="mixed"
    ).dt.normalize()
    lag = imputer.config.feature.max_day_lag
    context = anchors.loc[
        anchors["SIGHTING_DATE"].between(
            surface_date - pd.Timedelta(days=lag),
            surface_date + pd.Timedelta(days=lag),
        )
    ].copy()
    if context.empty:
        raise ValueError(f"No labeled context is available for ISO week {iso_week}")

    min_lat, max_lat, min_lon, max_lon = bounds
    canonical_graph = load_water_graph(
        surface_resolution,
        water_network_config_path,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        bbox_buffer_m=imputer.config.feature.max_radius_km * 1000.0,
    )
    grid = _canonical_water_cells(
        canonical_graph,
        context,
        parent_resolution=lookup.resolution,
        surface_resolution=surface_resolution,
        radius_km=imputer.config.feature.max_radius_km,
        bounds=bounds,
    )
    if grid.empty:
        raise ValueError("No water cells intersect the requested surface domain")

    seed_grid = grid.loc[grid["PARENT_H3"].isin(lookup.known_cells)].copy()
    seed_queries = seed_grid.assign(
        OBSERVATION_ID=lambda frame: [
            f"surface:{surface_date.date()}:{cell}" for cell in frame["H3_R8"]
        ],
        SIGHTING_DATE=surface_date,
        ECOTYPE_DETAIL="UNKNOWN",
    )
    seed_predictions = imputer.predict_queries(seed_queries, compute_stability=False)
    seed_predictions["H3_R8"] = seed_grid["H3_R8"].to_numpy()
    seed_predictions["LOCAL_BINARY_SUPPORT"] = (
        seed_predictions["LOCAL_SRKW_SUPPORT"]
        + seed_predictions["LOCAL_TRANSIENT_SUPPORT"]
    )
    seeds = seed_predictions.loc[
        seed_predictions["EVIDENCE_REGIME"].isin(SUPPORTED_SURFACE_REGIMES)
        & seed_predictions["MARINE_ROUTED_NEIGHBORS"].gt(0)
        & seed_predictions["LOCAL_BINARY_SUPPORT"].gt(0)
    ].copy()
    if seeds.empty:
        raise ValueError("The requested surface has no locally supported model seeds")

    cells = grid["H3_R8"].astype(str).tolist()
    positions = {cell: idx for idx, cell in enumerate(cells)}
    graph, distance_graph = _water_graph(canonical_graph, cells)
    graph_laplacian = laplacian(graph, normed=False).astype(np.float64).tocsr()
    center_spacing_km = h3.average_hexagon_edge_length(
        surface_resolution, unit="km"
    ) * np.sqrt(3)
    diffusion_time = 0.5 * (gaussian_sigma_km / center_spacing_km) ** 2

    seed_support = seeds["LOCAL_BINARY_SUPPORT"].to_numpy(dtype=float)
    support_cap = float(np.quantile(seed_support, 0.95))
    seed_support = np.clip(seed_support, 0.01, max(support_cap, 0.01))
    denominator = np.zeros(len(grid), dtype=np.float64)
    numerator = np.zeros(len(grid), dtype=np.float64)
    seed_indices = np.asarray([positions[cell] for cell in seeds["H3_R8"]], dtype=int)
    denominator[seed_indices] = seed_support
    numerator[seed_indices] = seed_support * seeds["P_SRKW"].to_numpy(dtype=float)
    diffused = expm_multiply(
        -diffusion_time * graph_laplacian,
        np.column_stack([numerator, denominator]),
    )
    diffused_numerator = diffused[:, 0]
    diffused_support = diffused[:, 1]
    distance_steps = dijkstra(
        distance_graph,
        directed=False,
        indices=seed_indices,
        min_only=True,
    )
    distance_km = distance_steps
    valid = (
        np.isfinite(distance_km)
        & (distance_km <= imputer.config.feature.max_radius_km)
        & (diffused_support > 1e-14)
    )
    surface = grid.loc[valid].copy()
    surface["SIGHTING_DATE"] = surface_date
    surface["P_SRKW_SURFACE"] = (
        diffused_numerator[valid] / diffused_support[valid]
    ).clip(0.0, 1.0)
    surface["P_TRANSIENT_SURFACE"] = 1.0 - surface["P_SRKW_SURFACE"]
    surface["DISTANCE_TO_MODEL_EVIDENCE_KM"] = distance_km[valid]
    scale = float(np.quantile(diffused_support[valid], 0.95))
    surface["SURFACE_EVIDENCE"] = np.clip(
        diffused_support[valid] / max(scale, 1e-12), 0.0, 1.0
    )
    surface["EVIDENCE_TIER"] = pd.cut(
        surface["SURFACE_EVIDENCE"],
        bins=[-np.inf, 0.08, 0.35, np.inf],
        labels=["Lower", "Medium", "Higher"],
    ).astype(str)

    seed_lookup = seeds.set_index("H3_R8")
    surface["P_SRKW_MODEL_SEED"] = surface["H3_R8"].map(seed_lookup["P_SRKW"])
    surface["MODEL_EVIDENCE_REGIME"] = surface["H3_R8"].map(
        seed_lookup["EVIDENCE_REGIME"]
    )
    surface["IS_MODEL_SEED"] = surface["P_SRKW_MODEL_SEED"].notna()
    surface["SURFACE_METHOD"] = np.where(
        surface["IS_MODEL_SEED"], "MODEL_SEED_GAUSSIAN_SMOOTHED", "WATER_GRAPH_GAUSSIAN"
    )
    return surface, week_start, week_end, surface_date


def make_probability_surface_map(
    surface: pd.DataFrame,
    sightings: pd.DataFrame,
    *,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    surface_date: pd.Timestamp,
    iso_week: int,
    year: int,
    maptiler_api_key: str | None = None,
    maptiler_style: str = "landscape-v4",
) -> go.Figure:
    """Render the H3 surface over MapTiler Landscape (or a keyless fallback)."""

    features: dict[str, dict] = {}
    for cell in surface["H3_R8"]:
        boundary = h3.cell_to_boundary(cell)
        ring = [[float(lon), float(lat)] for lat, lon in boundary]
        ring.append(ring[0])
        features[cell] = {
            "type": "Feature",
            "properties": {"h3_index": cell},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }

    figure = go.Figure()
    tier_opacity = {"Lower": 0.18, "Medium": 0.42, "Higher": 0.72}
    for tier in ("Lower", "Medium", "Higher"):
        layer = surface.loc[surface["EVIDENCE_TIER"].eq(tier)]
        if layer.empty:
            continue
        figure.add_trace(
            go.Choroplethmap(
                geojson={
                    "type": "FeatureCollection",
                    "features": [features[cell] for cell in layer["H3_R8"]],
                },
                featureidkey="properties.h3_index",
                locations=layer["H3_R8"],
                z=layer["P_SRKW_SURFACE"],
                zmin=0.0,
                zmax=1.0,
                colorscale=[
                    [0.0, "#D55E00"],
                    [0.5, "#F2F2F2"],
                    [1.0, "#0072B2"],
                ],
                marker_opacity=tier_opacity[tier],
                marker_line_width=0,
                customdata=layer[
                    [
                        "P_TRANSIENT_SURFACE",
                        "SURFACE_METHOD",
                        "DISTANCE_TO_MODEL_EVIDENCE_KM",
                        "SURFACE_EVIDENCE",
                        "P_SRKW_MODEL_SEED",
                    ]
                ],
                hovertemplate=(
                    "H3 %{location}<br>P(SRKW) %{z:.3f}<br>"
                    "P(Transient) %{customdata[0]:.3f}<br>"
                    "Method %{customdata[1]}<br>"
                    "Distance to model evidence %{customdata[2]:.1f} km<br>"
                    "Relative evidence %{customdata[3]:.3f}<br>"
                    "Seed P(SRKW) %{customdata[4]:.3f}<extra></extra>"
                ),
                showscale=tier == "Higher",
                colorbar={
                    "title": "P(SRKW)",
                    "tickvals": [0.0, 0.5, 1.0],
                    "ticktext": ["Transient", "Ambiguous", "SRKW"],
                    "x": 0.995,
                    "xanchor": "right",
                    "y": 0.46,
                    "len": 0.64,
                    "thickness": 16,
                },
                showlegend=False,
            )
        )

    dates = pd.to_datetime(sightings["SIGHTING_DATE"], format="mixed")
    week_sightings = sightings.loc[
        dates.between(week_start, week_end)
        & sightings["DISPLAY_CLASS"].isin(
            ["SRKW", "TRANSIENT", "SRKW_ASSIGNED", "TRANSIENT_ASSIGNED"]
        )
    ].copy()
    for display_class, points in week_sightings.groupby("DISPLAY_CLASS", observed=True):
        figure.add_trace(
            go.Scattermap(
                lat=points["LATITUDE"],
                lon=points["LONGITUDE"],
                mode="markers",
                marker={
                    "size": 8,
                    "color": MAP_COLORS[str(display_class)],
                    "opacity": 0.94,
                },
                text=points["OBSERVATION_ID"],
                customdata=points[["SIGHTING_DATE"]],
                hovertemplate=(
                    "%{text}<br>%{customdata[0]}<extra>"
                    + str(display_class)
                    + "</extra>"
                ),
                name=str(display_class).replace("_", " ").title(),
            )
        )

    map_layout: dict = {
        "center": {
            "lat": float(surface["LATITUDE"].median()),
            "lon": float(surface["LONGITUDE"].median()),
        },
        "zoom": 6.0,
    }
    if maptiler_api_key:
        map_layout.update(
            {
                "style": "white-bg",
                "layers": [
                    {
                        "below": "traces",
                        "sourcetype": "raster",
                        "source": [
                            "https://api.maptiler.com/maps/"
                            f"{maptiler_style}/256/{{z}}/{{x}}/{{y}}.png"
                            f"?key={maptiler_api_key}"
                        ],
                        "sourceattribution": "© MapTiler © OpenStreetMap contributors",
                    }
                ],
            }
        )
    else:
        map_layout["style"] = "carto-positron"

    basemap_name = (
        "MapTiler Landscape v4" if maptiler_api_key else "Carto Positron fallback"
    )
    figure.update_layout(
        title=(
            f"Ecotype probability surface — ISO week {iso_week}, {year} "
            f"(midpoint {surface_date.date()}, H3 R8; {basemap_name})"
        ),
        map=map_layout,
        legend={
            "title": {"text": "Week sightings"},
            "orientation": "h",
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.82)",
        },
        margin={"l": 0, "r": 20, "t": 60, "b": 0},
        height=800,
    )
    return figure
