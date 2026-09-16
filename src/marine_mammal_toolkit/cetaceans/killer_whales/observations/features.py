from __future__ import annotations

from marine_mammal_toolkit.tools.observations.impute.components import FeatureBatch
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from marine_mammal_toolkit.tools.observations.impute.config import FeatureConfig
from marine_mammal_toolkit.tools.observations.impute.config import Regime
from marine_mammal_toolkit.tools.observations.impute.marine import MarineDistanceLookup

EARTH_RADIUS_KM = 6371.0088
MODEL_CLASSES = ("SRKW", "TRANSIENT", "OTHER")


def prepare_model_frame(frame: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    detail = out["ECOTYPE_DETAIL"].astype(str).str.upper()
    out["MODEL_CLASS"] = pd.NA
    out.loc[detail.eq("SRKW"), "MODEL_CLASS"] = "SRKW"
    out.loc[detail.eq("TRANSIENT"), "MODEL_CLASS"] = "TRANSIENT"
    out.loc[detail.isin(cfg.known_other_labels), "MODEL_CLASS"] = "OTHER"
    if "ANCHOR_WEIGHT" not in out:
        out["ANCHOR_WEIGHT"] = 1.0
    if "ENCOUNTER_ID" not in out:
        out["ENCOUNTER_ID"] = out["OBSERVATION_ID"].astype(str)
    out["ANCHOR_WEIGHT"] = pd.to_numeric(out["ANCHOR_WEIGHT"], errors="coerce").fillna(
        1.0
    )
    out["ANCHOR_WEIGHT"] = out["ANCHOR_WEIGHT"].clip(0.0, 1.0)
    out["SIGHTING_DATE"] = pd.to_datetime(
        out["SIGHTING_DATE"], format="mixed"
    ).dt.normalize()
    return out


def _spatial_grid(
    lat: np.ndarray, lon: np.ndarray, cell_km: float
) -> tuple[np.ndarray, np.ndarray]:
    y = lat * 110.574
    x = lon * 111.320 * np.cos(np.deg2rad(lat))
    return np.floor(x / cell_km).astype(np.int64), np.floor(y / cell_km).astype(
        np.int64
    )


def _entropy(shares: Iterable[float]) -> float:
    values = np.asarray(list(shares), dtype=float)
    values = values[(values > 0) & np.isfinite(values)]
    return float(-(values * np.log(values)).sum()) if len(values) else 0.0


class DateContextFeatureBuilder:
    """Build day-resolution spatial context features using complete radius queries."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.config.validate()
        self.marine_lookup: MarineDistanceLookup | None = None

    def _ensure_marine_lookup(self) -> None:
        """Load the canonical graph only when feature transformation needs it."""

        if self.marine_lookup is None:
            self.marine_lookup = MarineDistanceLookup(
                self.config.water_network_config_path,
                resolution=self.config.marine_h3_resolution,
                required_radius_km=self.config.max_radius_km,
            )

    def _windows(self, regime: Regime) -> tuple[tuple[str, int, int], ...]:
        windows = self.config.temporal_windows
        if regime == "operational_eod":
            windows = tuple(item for item in windows if item[2] <= 0)
        return windows

    def transform(
        self,
        targets: pd.DataFrame,
        anchors: pd.DataFrame,
        *,
        regime: Regime = "retrospective",
        exclude_self: bool = True,
        exclude_same_day: bool = False,
        exclude_future: bool = False,
        excluded_anchor_encounters: pd.Series | np.ndarray | None = None,
    ) -> FeatureBatch:
        if targets.empty:
            return FeatureBatch(
                pd.DataFrame(index=targets.index), pd.DataFrame(index=targets.index)
            )
        self._ensure_marine_lookup()
        required = {"OBSERVATION_ID", "SIGHTING_DATE", "LATITUDE", "LONGITUDE"}
        for name, frame in (("targets", targets), ("anchors", anchors)):
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{name} is missing columns: {missing}")

        anchors = prepare_model_frame(anchors, self.config)
        anchors = anchors.loc[
            anchors["MODEL_CLASS"].isin(MODEL_CLASSES) & anchors["ANCHOR_WEIGHT"].gt(0)
        ].copy()
        if anchors.empty:
            raise ValueError("No eligible SRKW, TRANSIENT, or known-OTHER anchors")

        targets = targets.copy()
        targets["SIGHTING_DATE"] = pd.to_datetime(
            targets["SIGHTING_DATE"], format="mixed"
        ).dt.normalize()
        excluded_encounters = (
            np.full(len(targets), None, dtype=object)
            if excluded_anchor_encounters is None
            else np.asarray(excluded_anchor_encounters, dtype=object)
        )
        if len(excluded_encounters) != len(targets):
            raise ValueError("excluded_anchor_encounters must align with targets")
        windows = self._windows(regime)
        window_names = [item[0] for item in windows]

        a_lat = anchors["LATITUDE"].to_numpy(dtype=float)
        a_lon = anchors["LONGITUDE"].to_numpy(dtype=float)
        a_rad = np.deg2rad(np.column_stack([a_lat, a_lon]))
        a_date = anchors["SIGHTING_DATE"].to_numpy(dtype="datetime64[D]")
        a_id = anchors["OBSERVATION_ID"].astype(str).to_numpy()
        a_encounter = anchors["ENCOUNTER_ID"].astype(str).to_numpy()
        a_class = anchors["MODEL_CLASS"].astype(str).to_numpy()
        a_quality = anchors["ANCHOR_WEIGHT"].to_numpy(dtype=float)
        a_h3 = (
            self.marine_lookup.cells_for_coordinates(a_lat, a_lon)
            if self.marine_lookup is not None
            else None
        )
        a_xcell, a_ycell = _spatial_grid(a_lat, a_lon, self.config.support_unit_km)
        a_date_ord = a_date.astype(np.int64)

        # Query time before space. A single all-years BallTree returns thousands
        # of geographically nearby but temporally irrelevant anchors per target.
        # Small reusable date-block trees preserve exact day filtering while
        # avoiding most of that work.
        date_block_days = min(7, max(1, self.config.max_day_lag + 1))
        a_date_block = np.floor_divide(a_date_ord, date_block_days)
        block_trees: dict[int, tuple[np.ndarray, BallTree]] = {}
        for block in np.unique(a_date_block):
            positions = np.flatnonzero(a_date_block == block)
            block_trees[int(block)] = (
                positions,
                BallTree(a_rad[positions], metric="haversine"),
            )
        radius_rad = self.config.max_radius_km / EARTH_RADIUS_KM

        feature_rows: list[dict[str, float]] = []
        meta_rows: list[dict[str, object]] = []

        for start in range(0, len(targets), self.config.query_chunk_size):
            chunk = targets.iloc[start : start + self.config.query_chunk_size]
            q_lat = chunk["LATITUDE"].to_numpy(dtype=float)
            q_lon = chunk["LONGITUDE"].to_numpy(dtype=float)
            q_rad = np.deg2rad(np.column_stack([q_lat, q_lon]))
            q_h3 = (
                self.marine_lookup.cells_for_coordinates(q_lat, q_lon)
                if self.marine_lookup is not None
                else None
            )
            q_id = chunk["OBSERVATION_ID"].astype(str).to_numpy()
            q_date = chunk["SIGHTING_DATE"].to_numpy(dtype="datetime64[D]")
            q_date_ord = q_date.astype(np.int64)
            low_block = np.floor_divide(
                q_date_ord - self.config.max_day_lag, date_block_days
            )
            high_day = (
                q_date_ord
                if regime == "operational_eod"
                else q_date_ord + self.config.max_day_lag
            )
            high_block = np.floor_divide(high_day, date_block_days)
            needed: dict[int, list[int]] = {}
            for position, (lo, hi) in enumerate(
                zip(low_block, high_block, strict=True)
            ):
                for block in range(int(lo), int(hi) + 1):
                    if block in block_trees:
                        needed.setdefault(block, []).append(position)

            index_parts: list[list[np.ndarray]] = [[] for _ in range(len(chunk))]
            distance_parts: list[list[np.ndarray]] = [[] for _ in range(len(chunk))]
            for block, target_positions_list in needed.items():
                target_positions = np.asarray(target_positions_list, dtype=int)
                anchor_positions, block_tree = block_trees[block]
                local_indices, block_distances = block_tree.query_radius(
                    q_rad[target_positions],
                    r=radius_rad,
                    return_distance=True,
                    sort_results=False,
                )
                for local_position, target_position in enumerate(target_positions):
                    index_parts[int(target_position)].append(
                        anchor_positions[
                            np.asarray(local_indices[local_position], dtype=int)
                        ]
                    )
                    distance_parts[int(target_position)].append(
                        np.asarray(block_distances[local_position], dtype=float)
                    )
            indices = [
                np.concatenate(parts) if parts else np.asarray([], dtype=int)
                for parts in index_parts
            ]
            distances = [
                np.concatenate(parts) if parts else np.asarray([], dtype=float)
                for parts in distance_parts
            ]
            for offset in range(len(chunk)):
                neighbor_idx = np.asarray(indices[offset], dtype=int)
                neighbor_km = (
                    np.asarray(distances[offset], dtype=float) * EARTH_RADIUS_KM
                )
                if exclude_self and len(neighbor_idx):
                    keep = a_id[neighbor_idx] != q_id[offset]
                    neighbor_idx = neighbor_idx[keep]
                    neighbor_km = neighbor_km[keep]
                excluded_encounter = excluded_encounters[start + offset]
                if (
                    len(neighbor_idx)
                    and excluded_encounter is not None
                    and not pd.isna(excluded_encounter)
                ):
                    keep = a_encounter[neighbor_idx] != str(excluded_encounter)
                    neighbor_idx = neighbor_idx[keep]
                    neighbor_km = neighbor_km[keep]

                target_date = q_date[offset]
                if len(neighbor_idx):
                    lag = (
                        (a_date[neighbor_idx] - target_date)
                        .astype("timedelta64[D]")
                        .astype(int)
                    )
                    keep = np.abs(lag) <= self.config.max_day_lag
                    if regime == "operational_eod":
                        keep &= lag <= 0
                    if exclude_same_day:
                        keep &= lag != 0
                    if exclude_future:
                        keep &= lag <= 0
                    neighbor_idx = neighbor_idx[keep]
                    neighbor_km = neighbor_km[keep]
                    lag = lag[keep]
                else:
                    lag = np.asarray([], dtype=int)

                marine_candidate_neighbors = len(neighbor_idx)
                marine_routed_neighbors = 0
                marine_fallback_neighbors = 0
                marine_barrier_excluded_neighbors = 0
                marine_out_of_range_neighbors = 0
                marine_target_known = True
                marine_target_snap_km = 0.0
                target_h3 = None
                if self.marine_lookup is not None:
                    target_h3 = str(q_h3[offset])
                    marine_target_known = self.marine_lookup.is_known_cell(target_h3)
                    marine_target_snap_km = self.marine_lookup.snap_distance_km(
                        target_h3
                    )
                if self.marine_lookup is not None and len(neighbor_idx):
                    resolved_km, route_keep, fallback_mask = self.marine_lookup.resolve(
                        target_h3,
                        a_h3[neighbor_idx],
                        neighbor_km,
                        fallback_to_haversine=self.config.marine_fallback_to_haversine,
                    )
                    route_available = route_keep.copy()
                    within_radius = resolved_km <= self.config.max_radius_km
                    route_keep &= within_radius
                    marine_fallback_neighbors = int((fallback_mask & route_keep).sum())
                    marine_routed_neighbors = int((~fallback_mask & route_keep).sum())
                    marine_out_of_range_neighbors = int(
                        (route_available & ~within_radius).sum()
                    )
                    marine_barrier_excluded_neighbors = int((~route_available).sum())
                    neighbor_idx = neighbor_idx[route_keep]
                    neighbor_km = resolved_km[route_keep]
                    lag = lag[route_keep]
                elif self.marine_lookup is None:
                    marine_fallback_neighbors = marine_candidate_neighbors

                classes = (
                    a_class[neighbor_idx]
                    if len(neighbor_idx)
                    else np.asarray([], dtype=str)
                )
                nearest_evidence_encounter = (
                    str(a_encounter[neighbor_idx[int(np.argmin(neighbor_km))]])
                    if len(neighbor_idx)
                    else pd.NA
                )
                quality = (
                    a_quality[neighbor_idx]
                    if len(neighbor_idx)
                    else np.asarray([], dtype=float)
                )
                dist_weight = (
                    np.exp(-neighbor_km / self.config.distance_scale_km)
                    if len(neighbor_idx)
                    else np.asarray([], dtype=float)
                )
                day_weight = (
                    np.exp(-np.abs(lag) / self.config.time_scale_days)
                    if len(neighbor_idx)
                    else np.asarray([], dtype=float)
                )
                raw_weight = dist_weight * day_weight * quality

                row: dict[str, float] = {}
                class_window_support: dict[tuple[str, str], float] = {}
                class_window_units: dict[tuple[str, str], int] = {}

                for cls in MODEL_CLASSES:
                    class_mask = classes == cls
                    for window_name, lo, hi in windows:
                        mask = class_mask & (lag >= lo) & (lag <= hi)
                        idx_local = np.flatnonzero(mask)
                        prefix = f"{window_name}__{cls.lower()}"
                        if not len(idx_local):
                            row[f"{prefix}__support"] = 0.0
                            row[f"{prefix}__units"] = 0.0
                            row[f"{prefix}__effective_n"] = 0.0
                            row[f"{prefix}__nearest_km"] = np.nan
                            class_window_support[(cls, window_name)] = 0.0
                            class_window_units[(cls, window_name)] = 0
                            continue

                        actual_anchor_idx = neighbor_idx[idx_local]
                        weights = raw_weight[idx_local]
                        unit_keys = a_encounter[actual_anchor_idx]
                        _, inverse = np.unique(unit_keys, return_inverse=True)
                        unit_values = np.full(
                            int(inverse.max()) + 1, -np.inf, dtype=float
                        )
                        np.maximum.at(unit_values, inverse, weights)
                        support = float(unit_values.sum())
                        effective_n = (
                            float(support**2 / np.square(unit_values).sum())
                            if support > 0 and np.square(unit_values).sum() > 0
                            else 0.0
                        )
                        row[f"{prefix}__support"] = support
                        row[f"{prefix}__units"] = float(len(unit_values))
                        row[f"{prefix}__effective_n"] = effective_n
                        row[f"{prefix}__nearest_km"] = float(
                            neighbor_km[idx_local].min()
                        )
                        class_window_support[(cls, window_name)] = support
                        class_window_units[(cls, window_name)] = len(unit_values)

                same = {
                    cls: class_window_support.get((cls, "same_day"), 0.0)
                    for cls in MODEL_CLASSES
                }
                local = {
                    cls: float(
                        sum(
                            class_window_support.get((cls, window_name), 0.0)
                            for window_name in window_names
                        )
                    )
                    for cls in MODEL_CLASSES
                }
                local_units = {
                    cls: int(
                        sum(
                            class_window_units.get((cls, window_name), 0)
                            for window_name in window_names
                        )
                    )
                    for cls in MODEL_CLASSES
                }

                nearest_diagnostics: dict[str, float] = {}
                directional_support: dict[tuple[str, str], float] = {}
                movement_diagnostics: dict[str, float] = {}
                for cls in MODEL_CLASSES:
                    positions = np.flatnonzero(classes == cls)
                    cls_key = cls.lower()
                    if len(positions):
                        distance_position = positions[np.argmin(neighbor_km[positions])]
                        temporal_order = np.lexsort(
                            (neighbor_km[positions], np.abs(lag[positions]))
                        )
                        temporal_position = positions[temporal_order[0]]
                        nearest_diagnostics[f"nearest_km__{cls_key}"] = float(
                            neighbor_km[distance_position]
                        )
                        nearest_diagnostics[f"nearest_abs_day_lag__{cls_key}"] = float(
                            abs(lag[temporal_position])
                        )
                        nearest_diagnostics[f"nearest_day_lag__{cls_key}"] = float(
                            lag[temporal_position]
                        )
                    else:
                        nearest_diagnostics[f"nearest_km__{cls_key}"] = np.nan
                        nearest_diagnostics[f"nearest_abs_day_lag__{cls_key}"] = np.nan
                        nearest_diagnostics[f"nearest_day_lag__{cls_key}"] = np.nan
                    directional_support[(cls, "past")] = float(
                        sum(
                            class_window_support.get((cls, name), 0.0)
                            for name, _, hi in windows
                            if hi < 0
                        )
                    )
                    directional_support[(cls, "future")] = float(
                        sum(
                            class_window_support.get((cls, name), 0.0)
                            for name, lo, _ in windows
                            if lo > 0
                        )
                    )
                    row[f"past_total__{cls_key}"] = directional_support[(cls, "past")]
                    row[f"future_total__{cls_key}"] = directional_support[
                        (cls, "future")
                    ]
                    movement_positions = positions[lag[positions] != 0]
                    if len(movement_positions):
                        speeds = neighbor_km[movement_positions] / np.abs(
                            lag[movement_positions]
                        )
                        continuity_values = raw_weight[movement_positions] * np.exp(
                            -speeds / self.config.movement_speed_scale_km_per_day
                        )
                        feasible_values = raw_weight[movement_positions] * (
                            speeds <= self.config.maximum_feasible_km_per_day
                        )
                        encounter_keys = a_encounter[neighbor_idx[movement_positions]]
                        _, inverse = np.unique(encounter_keys, return_inverse=True)
                        continuity_units = np.zeros(int(inverse.max()) + 1, dtype=float)
                        feasible_units = np.zeros(int(inverse.max()) + 1, dtype=float)
                        np.maximum.at(continuity_units, inverse, continuity_values)
                        np.maximum.at(feasible_units, inverse, feasible_values)
                        movement_diagnostics[f"min_speed__{cls_key}"] = float(
                            speeds.min()
                        )
                        movement_diagnostics[f"median_speed__{cls_key}"] = float(
                            np.median(speeds)
                        )
                        movement_diagnostics[f"continuity__{cls_key}"] = float(
                            continuity_units.sum()
                        )
                        movement_diagnostics[f"feasible_support__{cls_key}"] = float(
                            feasible_units.sum()
                        )
                    else:
                        movement_diagnostics[f"min_speed__{cls_key}"] = np.nan
                        movement_diagnostics[f"median_speed__{cls_key}"] = np.nan
                        movement_diagnostics[f"continuity__{cls_key}"] = 0.0
                        movement_diagnostics[f"feasible_support__{cls_key}"] = 0.0
                    row[f"min_implied_speed_km_day__{cls_key}"] = movement_diagnostics[
                        f"min_speed__{cls_key}"
                    ]
                    row[f"median_implied_speed_km_day__{cls_key}"] = (
                        movement_diagnostics[f"median_speed__{cls_key}"]
                    )
                    row[f"movement_continuity__{cls_key}"] = movement_diagnostics[
                        f"continuity__{cls_key}"
                    ]
                    row[f"movement_feasible_support__{cls_key}"] = movement_diagnostics[
                        f"feasible_support__{cls_key}"
                    ]
                    row[f"past_future_bridge__{cls_key}"] = float(
                        np.sqrt(
                            directional_support[(cls, "past")]
                            * directional_support[(cls, "future")]
                        )
                    )
                    row.update(nearest_diagnostics)

                for cls in MODEL_CLASSES:
                    row[f"same_day_total__{cls.lower()}"] = same[cls]
                    row[f"local_total__{cls.lower()}"] = local[cls]
                    row[f"local_units__{cls.lower()}"] = float(local_units[cls])

                same_total = sum(same.values())
                local_binary_total = local["SRKW"] + local["TRANSIENT"]
                same_binary_total = same["SRKW"] + same["TRANSIENT"]
                srkw_same_share = (
                    same["SRKW"] / same_binary_total if same_binary_total else 0.5
                )
                srkw_local_share = (
                    local["SRKW"] / local_binary_total if local_binary_total else 0.5
                )
                row["same_day_binary_support"] = same_binary_total
                row["same_day_total_support"] = same_total
                row["same_day_srkw_share"] = srkw_same_share
                row["same_day_class_entropy"] = _entropy(
                    [same[cls] / same_total for cls in MODEL_CLASSES]
                    if same_total
                    else []
                )
                row["local_binary_support"] = local_binary_total
                row["local_srkw_share"] = srkw_local_share
                row["local_other_support"] = local["OTHER"]
                row["local_binary_support_margin"] = abs(
                    local["SRKW"] - local["TRANSIENT"]
                )
                row["local_binary_support_log_ratio"] = float(
                    np.log((local["SRKW"] + 1e-9) / (local["TRANSIENT"] + 1e-9))
                )
                row["movement_continuity_log_ratio"] = float(
                    np.log(
                        (movement_diagnostics["continuity__srkw"] + 1e-9)
                        / (movement_diagnostics["continuity__transient"] + 1e-9)
                    )
                )
                row["anchor_neighbors_in_radius"] = float(len(neighbor_idx))
                resolved_neighbors = marine_routed_neighbors + marine_fallback_neighbors
                row["marine_routed_neighbors"] = float(marine_routed_neighbors)
                row["marine_fallback_neighbors"] = float(marine_fallback_neighbors)
                row["marine_barrier_excluded_neighbors"] = float(
                    marine_barrier_excluded_neighbors
                )
                row["marine_lookup_coverage"] = (
                    marine_routed_neighbors / resolved_neighbors
                    if resolved_neighbors
                    else np.nan
                )
                row["marine_target_known"] = float(marine_target_known)
                row["marine_target_snap_km"] = marine_target_snap_km

                sighting_date = pd.Timestamp(target_date)
                doy = sighting_date.dayofyear
                row["latitude"] = float(q_lat[offset])
                row["longitude"] = float(q_lon[offset])
                row["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
                row["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))

                # The final known-OTHER veto is applied by the model policy.
                other_veto = False
                if self.marine_lookup is not None and not marine_target_known:
                    regime_name = "MARINE_LOOKUP_MISSING"
                elif (
                    self.marine_lookup is not None
                    and np.isfinite(marine_target_snap_km)
                    and marine_target_snap_km > self.config.max_target_snap_km
                ):
                    regime_name = "LOCATION_OFF_WATER"
                elif (
                    marine_candidate_neighbors > 0
                    and resolved_neighbors == 0
                    and marine_out_of_range_neighbors > 0
                ):
                    regime_name = "MARINE_OUT_OF_RANGE"
                elif marine_candidate_neighbors > 0 and resolved_neighbors == 0:
                    regime_name = "WATER_BLOCKED"
                elif same["OTHER"] > 0 and same_binary_total == 0:
                    regime_name = "OTHER_DOMINANT"
                elif same_binary_total > 0:
                    if same_binary_total < self.config.strong_local_support_floor:
                        regime_name = "WEAK_LOCAL"
                    else:
                        winner_share = max(srkw_same_share, 1 - srkw_same_share)
                        regime_name = (
                            "SAME_DAY_CONFLICT" if winner_share < 0.75 else "SAME_DAY"
                        )
                elif local_binary_total > 0:
                    if local_binary_total < self.config.strong_local_support_floor:
                        regime_name = "WEAK_LOCAL"
                    else:
                        winner_share = max(srkw_local_share, 1 - srkw_local_share)
                        regime_name = (
                            "LAGGED_CONFLICT" if winner_share < 0.65 else "LAGGED"
                        )
                elif marine_candidate_neighbors == 0:
                    regime_name = "NO_LOCAL_CANDIDATES"
                else:
                    regime_name = "OTHER_CONTEXT_ONLY"

                feature_rows.append(row)
                meta_rows.append(
                    {
                        "OBSERVATION_ID": q_id[offset],
                        "NEAREST_EVIDENCE_ENCOUNTER_ID": nearest_evidence_encounter,
                        "EVIDENCE_REGIME": regime_name,
                        "SAME_DAY_OTHER_SUPPORT": same["OTHER"],
                        "SAME_DAY_SRKW_SUPPORT": same["SRKW"],
                        "SAME_DAY_TRANSIENT_SUPPORT": same["TRANSIENT"],
                        "LOCAL_SRKW_SUPPORT": local["SRKW"],
                        "LOCAL_TRANSIENT_SUPPORT": local["TRANSIENT"],
                        "LOCAL_OTHER_SUPPORT": local["OTHER"],
                        "PAST_SRKW_SUPPORT": directional_support[("SRKW", "past")],
                        "PAST_TRANSIENT_SUPPORT": directional_support[
                            ("TRANSIENT", "past")
                        ],
                        "FUTURE_SRKW_SUPPORT": directional_support[("SRKW", "future")],
                        "FUTURE_TRANSIENT_SUPPORT": directional_support[
                            ("TRANSIENT", "future")
                        ],
                        "NEAREST_SRKW_KM": nearest_diagnostics["nearest_km__srkw"],
                        "NEAREST_TRANSIENT_KM": nearest_diagnostics[
                            "nearest_km__transient"
                        ],
                        "NEAREST_SRKW_DAY_LAG": nearest_diagnostics[
                            "nearest_day_lag__srkw"
                        ],
                        "NEAREST_TRANSIENT_DAY_LAG": nearest_diagnostics[
                            "nearest_day_lag__transient"
                        ],
                        "LOCAL_BINARY_SUPPORT_MARGIN": abs(
                            local["SRKW"] - local["TRANSIENT"]
                        ),
                        "LOCAL_BINARY_SUPPORT_LOG_RATIO": row[
                            "local_binary_support_log_ratio"
                        ],
                        "MIN_SRKW_IMPLIED_SPEED_KM_DAY": movement_diagnostics[
                            "min_speed__srkw"
                        ],
                        "MIN_TRANSIENT_IMPLIED_SPEED_KM_DAY": movement_diagnostics[
                            "min_speed__transient"
                        ],
                        "SRKW_MOVEMENT_CONTINUITY": movement_diagnostics[
                            "continuity__srkw"
                        ],
                        "TRANSIENT_MOVEMENT_CONTINUITY": movement_diagnostics[
                            "continuity__transient"
                        ],
                        "DISTANCE_METHOD": (
                            "MARINE_LOOKUP"
                            if self.marine_lookup is not None
                            else "HAVERSINE"
                        ),
                        "MARINE_CANDIDATE_NEIGHBORS": marine_candidate_neighbors,
                        "MARINE_ROUTED_NEIGHBORS": marine_routed_neighbors,
                        "MARINE_FALLBACK_NEIGHBORS": marine_fallback_neighbors,
                        "MARINE_BARRIER_EXCLUDED_NEIGHBORS": (
                            marine_barrier_excluded_neighbors
                        ),
                        "MARINE_OUT_OF_RANGE_NEIGHBORS": (
                            marine_out_of_range_neighbors
                        ),
                        "MARINE_LOOKUP_COVERAGE": row["marine_lookup_coverage"],
                        "MARINE_TARGET_KNOWN": marine_target_known,
                        "MARINE_TARGET_SNAP_KM": marine_target_snap_km,
                        "_OTHER_VETO_HINT": bool(other_veto),
                    }
                )

        X = pd.DataFrame(feature_rows, index=targets.index)
        meta = pd.DataFrame(meta_rows, index=targets.index)
        # Keep a deterministic feature order across folds and persisted models.
        X = X.reindex(sorted(X.columns), axis=1)
        return FeatureBatch(X=X, meta=meta)
