"""Learned marine space-time transport experiments for ecotype imputation.

This module is intentionally notebook-local and research-only.  It reconstructs
the current encounter-held-out folds, removes every test-fold encounter from the
anchor pool, computes several bounded marine distance/time kernels in one pass,
and evaluates transparent and regularized learned transport challengers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marine_mammal_toolkit.cetaceans.killer_whales.observations.features import EARTH_RADIUS_KM
from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputer import KillerWhaleImputer as SelectiveDateContextImputer

from experiment_support import (
    RANDOM_SEED,
    ReleasePaths,
    _encounter_reference,
    _paired_bootstrap,
    _poststratification_weights,
    _source_metric_table,
    binary_metrics,
    resolve_release_paths,
    write_json,
)

TRANSPORT_CLASSES = ("SRKW", "TRANSIENT", "OTHER")
TRANSPORT_DIRECTIONS = ("same", "past", "future")
MAX_RADIUS_KM = 40.0
MAX_DAY_LAG = 14


@dataclass(frozen=True)
class KernelSpec:
    distance_scale_km: float
    time_scale_days: float
    speed_scale_km_day: float | None = None

    @property
    def name(self) -> str:
        speed = "plain" if self.speed_scale_km_day is None else f"v{self.speed_scale_km_day:g}"
        return f"d{self.distance_scale_km:g}_t{self.time_scale_days:g}_{speed}"


def transport_kernel_grid() -> tuple[KernelSpec, ...]:
    return tuple(
        KernelSpec(distance_scale, time_scale, speed_scale)
        for distance_scale in (5.0, 10.0, 20.0, 40.0)
        for time_scale in (2.0, 4.0, 8.0, 14.0)
        for speed_scale in (None, 50.0)
    )


def _water_graph_sha256(graph: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(graph.resolution)).encode("utf-8"))
    digest.update(str(graph.water_mask_version).encode("utf-8"))
    digest.update(str(graph.spatial_support_version).encode("utf-8"))
    for values in (graph.cells, graph.offsets, graph.neighbors, graph.weights_m):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _date_block_trees(
    anchor_dates: np.ndarray,
    anchor_radians: np.ndarray,
) -> dict[int, tuple[np.ndarray, BallTree]]:
    block_days = 7
    date_blocks = np.floor_divide(anchor_dates.astype(np.int64), block_days)
    result: dict[int, tuple[np.ndarray, BallTree]] = {}
    for block in np.unique(date_blocks):
        positions = np.flatnonzero(date_blocks == block)
        result[int(block)] = (
            positions,
            BallTree(anchor_radians[positions], metric="haversine"),
        )
    return result


def _unit_maximum_sum(
    weights: np.ndarray,
    encounter_ids: np.ndarray,
) -> np.ndarray:
    if len(encounter_ids) == 0:
        return np.zeros(weights.shape[1], dtype=float)
    _, inverse = np.unique(encounter_ids, return_inverse=True)
    unit_maximum = np.zeros((int(inverse.max()) + 1, weights.shape[1]), dtype=float)
    for kernel_index in range(weights.shape[1]):
        np.maximum.at(unit_maximum[:, kernel_index], inverse, weights[:, kernel_index])
    return unit_maximum.sum(axis=0)


def _kernel_weights(
    distance_km: np.ndarray,
    lag_days: np.ndarray,
    quality: np.ndarray,
    kernels: Sequence[KernelSpec],
) -> np.ndarray:
    distances = distance_km[:, None]
    absolute_lag = np.abs(lag_days).astype(float)[:, None]
    distance_scales = np.asarray([item.distance_scale_km for item in kernels])[None, :]
    time_scales = np.asarray([item.time_scale_days for item in kernels])[None, :]
    exponent = -(distances / distance_scales) - (absolute_lag / time_scales)
    speed = np.divide(
        distances,
        absolute_lag,
        out=np.zeros_like(distances, dtype=float),
        where=absolute_lag > 0,
    )
    for index, kernel in enumerate(kernels):
        if kernel.speed_scale_km_day is not None:
            exponent[:, index] -= speed[:, 0] / kernel.speed_scale_km_day
    return np.exp(exponent) * quality[:, None]


def build_transport_support(
    targets: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    marine_lookup: Any,
    kernels: Sequence[KernelSpec],
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Build multi-kernel support with per-target encounter exclusion.

    ``anchors`` must already exclude every encounter in the outer test fold.
    Each target's own encounter is removed again, which prevents self-support
    for the outer-training rows.
    """

    anchors = anchors.loc[
        anchors["MODEL_CLASS"].isin(TRANSPORT_CLASSES)
        & anchors["ANCHOR_WEIGHT"].gt(0)
        & ~anchors["ENCOUNTER_LABEL_CONFLICT"].fillna(False)
    ].copy()
    anchors["SIGHTING_DATE"] = pd.to_datetime(anchors["SIGHTING_DATE"]).dt.normalize()
    targets = targets.copy()
    targets["SIGHTING_DATE"] = pd.to_datetime(targets["SIGHTING_DATE"]).dt.normalize()

    anchor_latitude = anchors["LATITUDE"].to_numpy(float)
    anchor_longitude = anchors["LONGITUDE"].to_numpy(float)
    anchor_radians = np.deg2rad(np.column_stack([anchor_latitude, anchor_longitude]))
    anchor_dates = anchors["SIGHTING_DATE"].to_numpy(dtype="datetime64[D]")
    anchor_ids = anchors["OBSERVATION_ID"].astype(str).to_numpy()
    anchor_encounters = anchors["ENCOUNTER_ID"].astype(str).to_numpy()
    anchor_classes = anchors["MODEL_CLASS"].astype(str).to_numpy()
    anchor_quality = anchors["ANCHOR_WEIGHT"].to_numpy(float)
    anchor_h3 = marine_lookup.cells_for_coordinates(anchor_latitude, anchor_longitude)
    block_trees = _date_block_trees(anchor_dates, anchor_radians)

    kernel_names = [item.name for item in kernels]
    rows: list[dict[str, float]] = []
    candidate_neighbor_count = 0
    routed_neighbor_count = 0
    blocked_neighbor_count = 0
    targets_without_routed_evidence = 0
    radius_radians = MAX_RADIUS_KM / EARTH_RADIUS_KM
    block_days = 7

    for target in targets.itertuples(index=False):
        target_latitude = float(target.LATITUDE)
        target_longitude = float(target.LONGITUDE)
        target_radians = np.deg2rad([[target_latitude, target_longitude]])
        target_date = np.datetime64(pd.Timestamp(target.SIGHTING_DATE).date(), "D")
        target_date_ordinal = int(target_date.astype(np.int64))
        low_block = (target_date_ordinal - MAX_DAY_LAG) // block_days
        high_block = (target_date_ordinal + MAX_DAY_LAG) // block_days
        index_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []
        for block in range(low_block, high_block + 1):
            if block not in block_trees:
                continue
            positions, tree = block_trees[block]
            local_indices, local_distances = tree.query_radius(
                target_radians,
                r=radius_radians,
                return_distance=True,
                sort_results=False,
            )
            if len(local_indices[0]):
                index_parts.append(positions[np.asarray(local_indices[0], dtype=int)])
                distance_parts.append(np.asarray(local_distances[0], dtype=float))
        neighbor_indices = np.concatenate(index_parts) if index_parts else np.asarray([], dtype=int)
        haversine_km = (
            np.concatenate(distance_parts) * EARTH_RADIUS_KM
            if distance_parts
            else np.asarray([], dtype=float)
        )
        if len(neighbor_indices):
            lag_days = (
                (anchor_dates[neighbor_indices] - target_date).astype("timedelta64[D]").astype(int)
            )
            keep = (
                (np.abs(lag_days) <= MAX_DAY_LAG)
                & (anchor_ids[neighbor_indices] != str(target.OBSERVATION_ID))
                & (anchor_encounters[neighbor_indices] != str(target.ENCOUNTER_ID))
            )
            neighbor_indices = neighbor_indices[keep]
            haversine_km = haversine_km[keep]
            lag_days = lag_days[keep]
        else:
            lag_days = np.asarray([], dtype=int)
        candidate_neighbor_count += len(neighbor_indices)

        target_h3 = marine_lookup.cells_for_coordinates(
            np.asarray([target_latitude]), np.asarray([target_longitude])
        )[0]
        if len(neighbor_indices):
            distance_km, route_keep, _fallback = marine_lookup.resolve(
                str(target_h3),
                anchor_h3[neighbor_indices],
                haversine_km,
                fallback_to_haversine=False,
            )
            blocked_neighbor_count += int((~route_keep).sum())
            neighbor_indices = neighbor_indices[route_keep]
            distance_km = distance_km[route_keep]
            lag_days = lag_days[route_keep]
        else:
            distance_km = np.asarray([], dtype=float)
        routed_neighbor_count += len(neighbor_indices)
        if len(neighbor_indices) == 0:
            targets_without_routed_evidence += 1

        row: dict[str, float] = {}
        if len(neighbor_indices):
            weights = _kernel_weights(
                distance_km,
                lag_days,
                anchor_quality[neighbor_indices],
                kernels,
            )
            classes = anchor_classes[neighbor_indices]
            encounters = anchor_encounters[neighbor_indices]
        else:
            weights = np.zeros((0, len(kernels)), dtype=float)
            classes = np.asarray([], dtype=str)
            encounters = np.asarray([], dtype=str)

        direction_masks = {
            "same": lag_days == 0,
            "past": lag_days < 0,
            "future": lag_days > 0,
        }
        for class_name in TRANSPORT_CLASSES:
            class_mask = classes == class_name
            for direction, direction_mask in direction_masks.items():
                selected = class_mask & direction_mask
                support = _unit_maximum_sum(weights[selected], encounters[selected])
                for kernel_name, value in zip(kernel_names, support, strict=True):
                    row[f"{kernel_name}__{class_name.lower()}__{direction}"] = float(value)
        rows.append(row)

    frame = pd.DataFrame(rows, index=targets.index).fillna(0.0)
    frame = frame.reindex(sorted(frame.columns), axis=1)
    diagnostics = {
        "target_n": len(targets),
        "anchor_n": len(anchors),
        "candidate_neighbor_n": candidate_neighbor_count,
        "routed_neighbor_n": routed_neighbor_count,
        "blocked_neighbor_n": blocked_neighbor_count,
        "target_without_routed_evidence_n": targets_without_routed_evidence,
        "target_without_routed_evidence_rate": (
            targets_without_routed_evidence / len(targets) if len(targets) else 0.0
        ),
    }
    return frame, diagnostics


def _support(
    features: pd.DataFrame,
    kernel_name: str,
    class_name: str,
    directions: Iterable[str] = TRANSPORT_DIRECTIONS,
) -> np.ndarray:
    columns = [f"{kernel_name}__{class_name.lower()}__{direction}" for direction in directions]
    return features[columns].sum(axis=1).to_numpy(float)


def _direct_probability(
    features: pd.DataFrame,
    kernel_name: str,
    *,
    prevalence: float,
    prior_strength: float,
    directions: Iterable[str] = TRANSPORT_DIRECTIONS,
) -> np.ndarray:
    srkw = _support(features, kernel_name, "SRKW", directions)
    transient = _support(features, kernel_name, "TRANSIENT", directions)
    return (srkw + prior_strength * prevalence) / (srkw + transient + prior_strength)


def _transport_design(
    features: pd.DataFrame,
    kernels: Sequence[KernelSpec],
    *,
    directions: Sequence[str],
) -> pd.DataFrame:
    result: dict[str, np.ndarray] = {}
    epsilon = 1e-6
    for kernel in kernels:
        for direction in directions:
            srkw = features[f"{kernel.name}__srkw__{direction}"].to_numpy(float)
            transient = features[f"{kernel.name}__transient__{direction}"].to_numpy(float)
            other = features[f"{kernel.name}__other__{direction}"].to_numpy(float)
            result[f"{kernel.name}__{direction}__log_ratio"] = np.log(
                (srkw + epsilon) / (transient + epsilon)
            )
            result[f"{kernel.name}__{direction}__binary_total"] = np.log1p(srkw + transient)
            result[f"{kernel.name}__{direction}__other"] = np.log1p(other)
    return pd.DataFrame(result, index=features.index)


def _new_sparse_logistic(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    l1_ratio=1.0,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _select_single_kernel(
    train_features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    kernels: Sequence[KernelSpec],
    *,
    prevalence: float,
) -> tuple[KernelSpec, float, float]:
    candidates: list[tuple[float, str, float, KernelSpec]] = []
    for kernel in kernels:
        for prior_strength in (0.05, 0.25, 1.0):
            probability = _direct_probability(
                train_features,
                kernel.name,
                prevalence=prevalence,
                prior_strength=prior_strength,
            )
            candidates.append(
                (
                    float(log_loss(target, probability, sample_weight=weights, labels=[0, 1])),
                    kernel.name,
                    prior_strength,
                    kernel,
                )
            )
    score, _name, prior_strength, kernel = min(candidates, key=lambda item: item[:3])
    return kernel, prior_strength, score


def _select_sparse_logistic(
    design: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 300)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for c_value in (0.002, 0.01, 0.05, 0.2):
        losses: list[float] = []
        crossfit_probability = np.full(len(target), np.nan, dtype=float)
        for fit_indices, score_indices in splitter.split(design, target, groups):
            model = _new_sparse_logistic(c_value)
            model.fit(
                design.iloc[fit_indices],
                target[fit_indices],
                model__sample_weight=weights[fit_indices],
            )
            probability = model.predict_proba(design.iloc[score_indices])[:, 1]
            crossfit_probability[score_indices] = probability
            losses.append(
                float(
                    log_loss(
                        target[score_indices],
                        probability,
                        sample_weight=weights[score_indices],
                        labels=[0, 1],
                    )
                )
            )
        if not np.isfinite(crossfit_probability).all():
            raise ValueError("Sparse transport tuning left rows without predictions")
        candidates.append((float(np.mean(losses)), c_value, crossfit_probability))
    score, c_value, probability = min(candidates, key=lambda item: (item[0], item[1]))
    return c_value, score, probability


def _select_constrained_blend(
    current_probability: np.ndarray,
    transport_probability: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_source_brier_degradation: float = 0.10,
) -> tuple[float, float, float]:
    candidates: list[tuple[float, float, float]] = []
    for current_weight in (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 1.0):
        probability = (
            current_weight * current_probability + (1 - current_weight) * transport_probability
        )
        degradations: list[float] = []
        for source_name in np.unique(source):
            mask = source == source_name
            if int(mask.sum()) < 50:
                continue
            baseline = float(
                brier_score_loss(
                    target[mask], current_probability[mask], sample_weight=weights[mask]
                )
            )
            challenger = float(
                brier_score_loss(target[mask], probability[mask], sample_weight=weights[mask])
            )
            degradations.append((challenger / baseline) - 1 if baseline > 0 else np.inf)
        maximum_degradation = max(degradations, default=0.0)
        if maximum_degradation <= maximum_source_brier_degradation + 1e-12:
            score = float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))
            candidates.append((score, current_weight, maximum_degradation))
    if not candidates:
        raise ValueError("The incumbent-only blend should always satisfy the source gate")
    score, current_weight, maximum_degradation = min(
        candidates, key=lambda item: (item[0], -item[1])
    )
    return current_weight, score, maximum_degradation


def _select_calibration_shrinkage(
    current_probability: np.ndarray,
    blend_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_source_brier_degradation: float = 0.10,
) -> tuple[float, float, float]:
    candidates: list[tuple[float, float, float]] = []
    for calibration_weight in (0.0, 0.01, 0.02, 0.025, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0):
        probability = (
            1 - calibration_weight
        ) * blend_probability + calibration_weight * calibrated_probability
        degradations: list[float] = []
        for source_name in np.unique(source):
            mask = source == source_name
            if int(mask.sum()) < 50:
                continue
            baseline = float(
                brier_score_loss(
                    target[mask], current_probability[mask], sample_weight=weights[mask]
                )
            )
            challenger = float(
                brier_score_loss(target[mask], probability[mask], sample_weight=weights[mask])
            )
            degradations.append((challenger / baseline) - 1 if baseline > 0 else np.inf)
        maximum_degradation = max(degradations, default=0.0)
        if maximum_degradation <= maximum_source_brier_degradation + 1e-12:
            score = float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))
            candidates.append((score, calibration_weight, maximum_degradation))
    if not candidates:
        raise ValueError("Zero calibration shrinkage should preserve the constrained blend")
    score, calibration_weight, maximum_degradation = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return calibration_weight, score, maximum_degradation


def _crossfit_platt_calibration(
    probability: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    score = logit(np.clip(probability, 1e-6, 1 - 1e-6))[:, None]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 400)
    crossfit_probability = np.full(len(target), np.nan, dtype=float)
    for fit_indices, score_indices in splitter.split(score, target, groups):
        calibrator = LogisticRegression(
            C=1_000_000.0,
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        calibrator.fit(
            score[fit_indices],
            target[fit_indices],
            sample_weight=weights[fit_indices],
        )
        crossfit_probability[score_indices] = calibrator.predict_proba(score[score_indices])[:, 1]
    if not np.isfinite(crossfit_probability).all():
        raise ValueError("Cross-fitted Platt calibration left rows without predictions")
    final_calibrator = LogisticRegression(
        C=1_000_000.0,
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    final_calibrator.fit(score, target, sample_weight=weights)
    return crossfit_probability, final_calibrator


def _top_coefficients(
    model: Pipeline,
    columns: Sequence[str],
    *,
    fold: int,
    variant: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    coefficient = model.named_steps["model"].coef_[0]
    order = np.argsort(np.abs(coefficient))[::-1][:limit]
    return [
        {
            "outer_fold": fold,
            "variant": variant,
            "feature": columns[index],
            "coefficient": float(coefficient[index]),
            "absolute_coefficient": float(abs(coefficient[index])),
        }
        for index in order
        if abs(coefficient[index]) > 1e-12
    ]


def run_transport_experiment(
    output_dir: Path,
    paths: ReleasePaths | None = None,
) -> dict[str, Any]:
    paths = paths or resolve_release_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    imputer = SelectiveDateContextImputer.load(paths.model_dir / "ecotype_imputer.joblib")
    anchors = imputer.anchors_.copy()
    kernels = transport_kernel_grid()
    oof = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    oof["MODEL_CLASS"] = oof["MODEL_CLASS"].str.upper()
    if oof["ENCOUNTER_ID"].duplicated().any():
        raise ValueError("Transport evaluation requires independent encounter representatives")

    reference = _encounter_reference(paths)
    binary_reference = reference.loc[reference["ECOTYPE_DETAIL"].isin(["SRKW", "TRANSIENT"])].copy()
    binary_reference["MODEL_CLASS"] = binary_reference["ECOTYPE_DETAIL"]
    weights = _poststratification_weights(oof, binary_reference, label_column="MODEL_CLASS")
    target = oof["Y_TRUE"].to_numpy(int)
    groups = oof["ENCOUNTER_ID"].astype(str).to_numpy()
    prevalence = float(np.average(target, weights=weights))
    outer = StratifiedGroupKFold(
        n_splits=int(imputer.config.model.n_splits),
        shuffle=True,
        random_state=int(imputer.config.model.random_state),
    )

    prediction_columns = {
        "fixed_kernel_direct": np.full(len(oof), np.nan),
        "learned_single_kernel": np.full(len(oof), np.nan),
        "learned_multiscale_retrospective": np.full(len(oof), np.nan),
        "learned_multiscale_past_only": np.full(len(oof), np.nan),
        "constrained_current_transport_blend": np.full(len(oof), np.nan),
        "calibrated_constrained_transport_blend": np.full(len(oof), np.nan),
        "gate_constrained_calibrated_blend": np.full(len(oof), np.nan),
    }
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    support_diagnostics: list[dict[str, Any]] = []
    fixed_kernel = next(
        item
        for item in kernels
        if item.distance_scale_km == 10
        and item.time_scale_days == 4
        and item.speed_scale_km_day is None
    )

    for fold, (train_indices, test_indices) in enumerate(outer.split(oof, target, groups), start=1):
        test_encounters = set(oof.iloc[test_indices]["ENCOUNTER_ID"].astype(str))
        nearest_leakage = (
            oof.iloc[test_indices]["NEAREST_EVIDENCE_ENCOUNTER_ID"]
            .astype(str)
            .isin(test_encounters)
        )
        if nearest_leakage.any():
            raise ValueError(
                "Configured encounter folds do not reproduce the persisted leakage-safe OOF split"
            )
        allowed_anchors = anchors.loc[
            ~anchors["ENCOUNTER_ID"].astype(str).isin(test_encounters)
        ].copy()
        fold_features, diagnostics = build_transport_support(
            oof,
            allowed_anchors,
            marine_lookup=imputer.feature_builder.marine_lookup,
            kernels=kernels,
        )
        support_diagnostics.append({"outer_fold": fold, **diagnostics})
        train_features = fold_features.iloc[train_indices]
        test_features = fold_features.iloc[test_indices]
        train_target = target[train_indices]
        train_weights = weights[train_indices]

        prediction_columns["fixed_kernel_direct"][test_indices] = _direct_probability(
            test_features,
            fixed_kernel.name,
            prevalence=prevalence,
            prior_strength=0.25,
        )
        selected_kernel, selected_prior, single_inner_loss = _select_single_kernel(
            train_features,
            train_target,
            train_weights,
            kernels,
            prevalence=prevalence,
        )
        prediction_columns["learned_single_kernel"][test_indices] = _direct_probability(
            test_features,
            selected_kernel.name,
            prevalence=prevalence,
            prior_strength=selected_prior,
        )

        variant_details: dict[str, dict[str, float]] = {}
        retrospective_train_oof: np.ndarray | None = None
        retrospective_test_probability: np.ndarray | None = None
        for variant, directions in (
            ("learned_multiscale_retrospective", ("same", "past", "future")),
            ("learned_multiscale_past_only", ("same", "past")),
        ):
            train_design = _transport_design(train_features, kernels, directions=directions)
            test_design = _transport_design(test_features, kernels, directions=directions)
            selected_c, inner_loss, train_oof_probability = _select_sparse_logistic(
                train_design,
                train_target,
                groups[train_indices],
                train_weights,
            )
            model = _new_sparse_logistic(selected_c)
            model.fit(
                train_design,
                train_target,
                model__sample_weight=train_weights,
            )
            test_probability = model.predict_proba(test_design)[:, 1]
            prediction_columns[variant][test_indices] = test_probability
            if variant == "learned_multiscale_retrospective":
                retrospective_train_oof = train_oof_probability
                retrospective_test_probability = test_probability
            coefficient_rows.extend(
                _top_coefficients(
                    model,
                    train_design.columns,
                    fold=fold,
                    variant=variant,
                )
            )
            variant_details[variant] = {"selected_c": selected_c, "inner_log_loss": inner_loss}

        if retrospective_train_oof is None or retrospective_test_probability is None:
            raise RuntimeError("Retrospective transport predictions were not produced")
        blend_weight, blend_inner_loss, blend_max_degradation = _select_constrained_blend(
            oof.iloc[train_indices]["P_SRKW"].to_numpy(float),
            retrospective_train_oof,
            train_target,
            oof.iloc[train_indices]["SOURCE"].astype(str).to_numpy(),
            train_weights,
        )
        train_blend_probability = (
            blend_weight * oof.iloc[train_indices]["P_SRKW"].to_numpy(float)
            + (1 - blend_weight) * retrospective_train_oof
        )
        test_blend_probability = (
            blend_weight * oof.iloc[test_indices]["P_SRKW"].to_numpy(float)
            + (1 - blend_weight) * retrospective_test_probability
        )
        prediction_columns["constrained_current_transport_blend"][
            test_indices
        ] = test_blend_probability
        train_calibrated_oof, calibration_model = _crossfit_platt_calibration(
            train_blend_probability,
            train_target,
            groups[train_indices],
            train_weights,
        )
        test_calibrated_probability = calibration_model.predict_proba(
            logit(np.clip(test_blend_probability, 1e-6, 1 - 1e-6))[:, None]
        )[:, 1]
        prediction_columns["calibrated_constrained_transport_blend"][
            test_indices
        ] = test_calibrated_probability
        calibration_weight, calibration_inner_loss, calibration_max_degradation = (
            _select_calibration_shrinkage(
                oof.iloc[train_indices]["P_SRKW"].to_numpy(float),
                train_blend_probability,
                train_calibrated_oof,
                train_target,
                oof.iloc[train_indices]["SOURCE"].astype(str).to_numpy(),
                train_weights,
            )
        )
        prediction_columns["gate_constrained_calibrated_blend"][test_indices] = (
            1 - calibration_weight
        ) * test_blend_probability + calibration_weight * test_calibrated_probability

        fold_rows.append(
            {
                "outer_fold": fold,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "test_start": oof.iloc[test_indices]["SIGHTING_DATE"].min(),
                "test_end": oof.iloc[test_indices]["SIGHTING_DATE"].max(),
                "selected_single_kernel": selected_kernel.name,
                "selected_single_prior_strength": selected_prior,
                "single_inner_log_loss": single_inner_loss,
                "retrospective_selected_c": variant_details["learned_multiscale_retrospective"][
                    "selected_c"
                ],
                "retrospective_inner_log_loss": variant_details["learned_multiscale_retrospective"][
                    "inner_log_loss"
                ],
                "past_only_selected_c": variant_details["learned_multiscale_past_only"][
                    "selected_c"
                ],
                "past_only_inner_log_loss": variant_details["learned_multiscale_past_only"][
                    "inner_log_loss"
                ],
                "blend_current_weight": blend_weight,
                "blend_inner_log_loss": blend_inner_loss,
                "blend_training_max_source_brier_degradation": blend_max_degradation,
                "blend_calibration_intercept": float(calibration_model.intercept_[0]),
                "blend_calibration_slope": float(calibration_model.coef_[0, 0]),
                "blend_calibration_weight": calibration_weight,
                "blend_calibration_inner_log_loss": calibration_inner_loss,
                "blend_calibration_training_max_source_brier_degradation": (
                    calibration_max_degradation
                ),
            }
        )

    for name, probability in prediction_columns.items():
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"{name} did not produce finite unit-interval probabilities")
        oof[name] = probability
    oof["current_model"] = oof["P_SRKW"].to_numpy(float)

    comparison = pd.DataFrame(
        [
            {"model": name, **binary_metrics(target, oof[name], weights)}
            for name in (
                "current_model",
                "fixed_kernel_direct",
                "learned_single_kernel",
                "learned_multiscale_past_only",
                "learned_multiscale_retrospective",
                "constrained_current_transport_blend",
                "calibrated_constrained_transport_blend",
                "gate_constrained_calibrated_blend",
            )
        ]
    )
    metric_export = oof.rename(
        columns={name: f"P__{name}" for name in ("current_model", *prediction_columns)}
    )
    source_metrics = _source_metric_table(
        metric_export,
        [f"P__{name}" for name in ("current_model", *prediction_columns)],
        weights,
    )
    source_gate_rows: list[dict[str, Any]] = []
    current_by_source = source_metrics.loc[
        source_metrics["model"].eq("P__current_model")
    ].set_index("source")
    for model_name in prediction_columns:
        challenger_by_source = source_metrics.loc[
            source_metrics["model"].eq(f"P__{model_name}")
        ].set_index("source")
        for source_name in current_by_source.index:
            current_brier = float(current_by_source.loc[source_name, "brier"])
            challenger_brier = float(challenger_by_source.loc[source_name, "brier"])
            independent_n = int(current_by_source.loc[source_name, "n"])
            relative_degradation = (
                challenger_brier / current_brier - 1 if current_brier > 0 else np.inf
            )
            source_gate_rows.append(
                {
                    "model": model_name,
                    "source": source_name,
                    "independent_n": independent_n,
                    "current_brier": current_brier,
                    "challenger_brier": challenger_brier,
                    "relative_brier_degradation": relative_degradation,
                    "required_gate": independent_n >= 100,
                    "passes_10pct_degradation_gate": (
                        independent_n < 100 or relative_degradation <= 0.10
                    ),
                }
            )
    source_gates = pd.DataFrame(source_gate_rows)
    source_gate_summary = (
        source_gates.loc[source_gates["required_gate"]]
        .groupby("model", as_index=False)
        .agg(
            required_source_n=("source", "nunique"),
            failed_source_n=(
                "passes_10pct_degradation_gate",
                lambda values: int((~values).sum()),
            ),
            maximum_relative_brier_degradation=("relative_brier_degradation", "max"),
        )
    )
    source_gate_summary["all_required_sources_pass"] = source_gate_summary["failed_source_n"].eq(0)
    eligible_names = set(
        source_gate_summary.loc[source_gate_summary["all_required_sources_pass"], "model"]
    )
    eligible_comparison = comparison.loc[comparison["model"].isin(eligible_names)].copy()
    recommended_challenger = (
        str(eligible_comparison.sort_values(["log_loss", "brier"]).iloc[0]["model"])
        if not eligible_comparison.empty
        else None
    )
    bootstrap = {
        name: _paired_bootstrap(
            target,
            oof["current_model"].to_numpy(float),
            oof[name].to_numpy(float),
            weights,
        )
        for name in prediction_columns
    }
    fold_selections = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics_frame = pd.DataFrame(support_diagnostics)
    prediction_export = oof[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "SOURCE",
            "MODEL_CLASS",
            "Y_TRUE",
            "current_model",
            *prediction_columns,
        ]
    ].copy()

    frames = {
        "transport_comparison": comparison,
        "transport_metrics_by_source": source_metrics,
        "transport_source_gates": source_gates,
        "transport_source_gate_summary": source_gate_summary,
        "transport_fold_selections": fold_selections,
        "transport_top_coefficients": coefficients,
        "transport_support_diagnostics": diagnostics_frame,
        "transport_oof_predictions": prediction_export,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "experiment": "learned marine space-time ecotype evidence transport",
            "release_id": paths.release_id,
            "snapshot_id": paths.snapshot_id,
            "release_manifest": paths.manifest_path,
            "model_sha256": imputer.training_summary_.get("MODEL_IDENTITY_SHA256"),
            "water_graph_sha256": _water_graph_sha256(imputer.feature_builder.marine_lookup.graph),
            "water_graph_resolution": int(imputer.feature_builder.marine_lookup.graph.resolution),
            "water_mask_version": (imputer.feature_builder.marine_lookup.graph.water_mask_version),
            "spatial_support_version": (
                imputer.feature_builder.marine_lookup.graph.spatial_support_version
            ),
            "outer_strategy": "reconstructed current encounter folds",
            "outer_folds": int(imputer.config.model.n_splits),
            "inner_folds": 3,
            "anchor_exclusion": "all outer-test encounters plus each target encounter",
            "maximum_radius_km": MAX_RADIUS_KM,
            "maximum_day_lag": MAX_DAY_LAG,
            "kernel_grid": [item.__dict__ | {"name": item.name} for item in kernels],
            "source_used_as_feature": False,
            "natural_prevalence_poststratification": "SOURCE x observed class encounter prevalence",
            "bootstrap_vs_current": bootstrap,
            "recommended_research_challenger": recommended_challenger,
            "recommended_challenger_rule": (
                "lowest natural-prevalence log loss among candidates passing every "
                "required source's 10 percent relative Brier degradation gate"
            ),
            "promotion_eligible": False,
            "promotion_blockers": [
                "binary SRKW-versus-Transient target remains closed-set",
                "evaluation uses the current capped encounter sample",
                "only three configured outer folds are available for exact incumbent comparison",
                "rolling-origin, repeated spatial, and leave-one-source-out transport tests remain required",
                "no blinded unknown-label audit sample exists",
            ],
            "outputs": [f"{name}.csv" for name in frames],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        **frames,
        "bootstrap": bootstrap,
        "manifest_path": manifest_path,
    }


__all__ = [
    "KernelSpec",
    "build_transport_support",
    "run_transport_experiment",
    "transport_kernel_grid",
]
