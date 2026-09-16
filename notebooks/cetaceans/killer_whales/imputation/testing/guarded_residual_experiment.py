"""Guarded residual challengers for retrospective ecotype imputation.

The experiment keeps the constrained current/transport blend as an offset,
learns only a bounded residual correction from season and a reduced physical
seascape panel, calibrates the score with regularized source/era/region effects,
and shrinks unreliable rows back toward the offset baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from marine_mammal_toolkit.tools._core.checksums import checksum_path
from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputer import KillerWhaleImputer as SelectiveDateContextImputer

from covariate_experiment import (
    _gate_table,
    _stratified_metrics,
    build_seasonal_features,
    load_static_seascape_features,
)
from experiment_support import (
    RANDOM_SEED,
    ReleasePaths,
    _encounter_reference,
    _paired_bootstrap,
    _poststratification_weights,
    binary_metrics,
    resolve_release_paths,
    write_json,
)
from transport_experiment import (
    _crossfit_platt_calibration,
    _new_sparse_logistic,
    _select_constrained_blend,
    _select_sparse_logistic,
    _transport_design,
    _water_graph_sha256,
    build_transport_support,
    transport_kernel_grid,
)

REDUCED_SEASCAPE_FEATURES = (
    "seascape__bathymetry__bathymetry_median",
    "seascape__bathymetry__bathymetry_std",
    "seascape__bathymetry__bathymetry_local_anomaly",
    "seascape__bathymetry__distance_to_isobath_50_m",
    "seascape__bathymetry__distance_to_isobath_200_m",
    "seascape__bathymetry__bathymetry_frac_0_10_m",
    "seascape__bathymetry__bathymetry_frac_over_200_m",
    "seascape__geomorphometry__slope_mean_ring_2",
    "seascape__geomorphometry__terrain_position_ring_2_z",
    "seascape__geomorphometry__local_relief_ring_2_m",
    "seascape__geomorphometry__vector_ruggedness_ring_2",
    "seascape__geomorphometry__positive_openness_deg",
    "seascape__geomorphometry__negative_openness_deg",
    "seascape__geomorphic_units__broad_terrain_position_z",
    "seascape__geomorphic_units__canyon_density",
    "seascape__geomorphic_units__distance_to_shelf_break_m",
    "seascape__geomorphic_units__distance_to_canyon_axis_m",
    "seascape__shoreline_proximity__shoreline_distance_m",
    "seascape__shoreline_proximity__water_network_distance_m",
    "seascape__shoreline_proximity__open_ocean_index",
    "seascape__exposure_enclosure__openness_to_ocean_index",
    "seascape__exposure_enclosure__enclosure_index",
    "seascape__exposure_enclosure__distance_to_open_water_m",
    "seascape__waterbody_morphometry__local_waterbody_width_m",
    "seascape__waterbody_morphometry__constriction_index",
    "seascape__waterbody_morphometry__distance_to_constricted_passage_m",
    "seascape__estuarine_connectivity__water_network_distance_to_estuary_m",
    "seascape__fluvial_connectivity__water_network_distance_to_fluvial_mouth_m",
    "seascape__fluvial_connectivity__fluvial_path_detour_ratio",
    "seascape__fluvial_connectivity__connected_upstream_drainage_area_km2",
    "seascape__benthic_substrate__substrate_hard_substrate_frac",
    "seascape__benthic_substrate__substrate_sand_frac",
    "seascape__benthic_substrate__substrate_mud_frac",
    "seascape__benthic_substrate__substrate_heterogeneity",
    "seascape__bottom_hardness__bottom_hardness_index",
)

SEASONAL_SEASCAPE_INTERACTIONS = (
    "seascape__bathymetry__bathymetry_median",
    "seascape__bathymetry__distance_to_isobath_200_m",
    "seascape__shoreline_proximity__water_network_distance_m",
    "seascape__shoreline_proximity__open_ocean_index",
    "seascape__waterbody_morphometry__local_waterbody_width_m",
    "seascape__estuarine_connectivity__water_network_distance_to_estuary_m",
    "seascape__fluvial_connectivity__water_network_distance_to_fluvial_mouth_m",
    "seascape__benthic_substrate__substrate_hard_substrate_frac",
)

CALIBRATION_CATEGORICAL_COLUMNS = (
    "SOURCE",
    "ERA",
    "REGION",
    "EVIDENCE_REGIME",
    "SEASCAPE_COVERAGE",
)


@dataclass
class OffsetResidualModel:
    preprocessor: Pipeline
    intercept: float
    coefficient: np.ndarray
    converged: bool

    def predict_probability(
        self,
        design: pd.DataFrame,
        baseline_probability: np.ndarray,
        *,
        maximum_correction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.preprocessor.transform(design)
        correction = self.intercept + transformed @ self.coefficient
        correction = np.clip(correction, -maximum_correction, maximum_correction)
        probability = expit(logit(np.clip(baseline_probability, 1e-6, 1 - 1e-6)) + correction)
        return probability, correction


@dataclass
class AdaptiveBlender:
    global_weight: float
    group_weights: dict[str, dict[str, float]]

    def weights(self, strata: pd.DataFrame) -> np.ndarray:
        result = np.full(len(strata), self.global_weight, dtype=float)
        for column, lookup in self.group_weights.items():
            values = strata[column].astype(str).to_numpy()
            local = np.asarray([lookup.get(value, 0.0) for value in values])
            result = np.minimum(result, local)
        return np.clip(result, 0.0, 1.0)

    def predict(
        self,
        baseline_probability: np.ndarray,
        challenger_probability: np.ndarray,
        strata: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        challenger_weight = self.weights(strata)
        probability = (
            1.0 - challenger_weight
        ) * baseline_probability + challenger_weight * challenger_probability
        return probability, challenger_weight


def build_reduced_residual_design(
    baseline_probability: np.ndarray,
    transport_probability: np.ndarray,
    transport_support: pd.DataFrame,
    seasonal: pd.DataFrame,
    seascape: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(set(REDUCED_SEASCAPE_FEATURES) - set(seascape.columns))
    if missing:
        raise ValueError(f"Reduced seascape panel is missing features: {missing}")
    selected_columns: list[str] = list(REDUCED_SEASCAPE_FEATURES)
    selected_products = sorted({column.split("__")[1] for column in selected_columns})
    for column in tuple(selected_columns):
        missing_column = f"{column}__missing"
        if missing_column in seascape.columns:
            selected_columns.append(missing_column)
    for product in selected_products:
        availability = f"seascape__{product}__cell_available"
        if availability in seascape.columns:
            selected_columns.append(availability)

    support_columns = [
        column
        for column in transport_support.columns
        if "__srkw__" in column or "__transient__" in column
    ]
    transport_logit = logit(np.clip(transport_probability, 1e-6, 1 - 1e-6))
    baseline_logit = logit(np.clip(baseline_probability, 1e-6, 1 - 1e-6))
    design = pd.DataFrame(
        {
            "evidence__transport_logit": transport_logit,
            "evidence__transport_baseline_disagreement": transport_logit - baseline_logit,
            "evidence__transport_has_support": (
                transport_support[support_columns].sum(axis=1).gt(0).astype(float).to_numpy()
            ),
            "evidence__transport_total_support": np.log1p(
                transport_support[support_columns].sum(axis=1).to_numpy(float)
            ),
        },
        index=transport_support.index,
    )
    design = pd.concat([design, seasonal, seascape[selected_columns]], axis=1)
    interaction_parts: dict[str, pd.Series | np.ndarray] = {}
    for seasonal_column in [
        column for column in seasonal.columns if "__sin_" in column or "__cos_" in column
    ]:
        interaction_parts[f"interaction__transport__{seasonal_column}"] = (
            transport_logit * seasonal[seasonal_column].to_numpy(float)
        )
    for seascape_column in SEASONAL_SEASCAPE_INTERACTIONS:
        for seasonal_column in ("season__sin_1", "season__cos_1"):
            interaction_parts[f"interaction__{seasonal_column}__{seascape_column}"] = (
                seasonal[seasonal_column] * seascape[seascape_column]
            )
    design = pd.concat([design, pd.DataFrame(interaction_parts, index=design.index)], axis=1)
    if design.columns.duplicated().any():
        raise ValueError("Reduced residual design contains duplicate feature names")
    return design


def _fit_offset_residual(
    design: pd.DataFrame,
    target: np.ndarray,
    baseline_probability: np.ndarray,
    weights: np.ndarray,
    *,
    l2_penalty: float,
) -> OffsetResidualModel:
    preprocessor = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    transformed = np.asarray(preprocessor.fit_transform(design), dtype=float)
    baseline_logit = logit(np.clip(baseline_probability, 1e-6, 1 - 1e-6))
    normalized_weights = weights / weights.sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficient = parameters[1:]
        linear = baseline_logit + intercept + transformed @ coefficient
        loss = float(
            np.sum(normalized_weights * (np.logaddexp(0.0, linear) - target * linear))
            + 0.5 * l2_penalty * np.square(coefficient).sum()
        )
        residual = normalized_weights * (expit(linear) - target)
        gradient = np.concatenate(
            [
                np.asarray([residual.sum()]),
                transformed.T @ residual + l2_penalty * coefficient,
            ]
        )
        return loss, gradient

    initial = np.zeros(transformed.shape[1] + 1, dtype=float)
    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise ValueError("Offset residual optimization produced non-finite parameters")
    return OffsetResidualModel(
        preprocessor=preprocessor,
        intercept=float(fit.x[0]),
        coefficient=np.asarray(fit.x[1:], dtype=float),
        converged=bool(fit.success),
    )


def _select_offset_residual(
    design: pd.DataFrame,
    target: np.ndarray,
    baseline_probability: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 600)
    candidates = [
        (l2_penalty, maximum_correction)
        for l2_penalty in (0.01, 0.1, 1.0, 10.0)
        for maximum_correction in (0.25, 0.5, 1.0, 2.0)
    ]
    predictions = {candidate: np.full(len(target), np.nan, dtype=float) for candidate in candidates}
    for fit_indices, score_indices in splitter.split(design, target, groups):
        for l2_penalty in sorted({candidate[0] for candidate in candidates}):
            model = _fit_offset_residual(
                design.iloc[fit_indices],
                target[fit_indices],
                baseline_probability[fit_indices],
                weights[fit_indices],
                l2_penalty=l2_penalty,
            )
            for maximum_correction in sorted({candidate[1] for candidate in candidates}):
                probability, _correction = model.predict_probability(
                    design.iloc[score_indices],
                    baseline_probability[score_indices],
                    maximum_correction=maximum_correction,
                )
                predictions[(l2_penalty, maximum_correction)][score_indices] = probability
    scored: list[tuple[float, float, float, np.ndarray]] = []
    for (l2_penalty, maximum_correction), probability in predictions.items():
        if not np.isfinite(probability).all():
            raise ValueError("Offset residual tuning left rows without predictions")
        score = float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))
        scored.append((score, l2_penalty, maximum_correction, probability))
    score, l2_penalty, maximum_correction, probability = min(
        scored, key=lambda item: (item[0], item[1], item[2])
    )
    return l2_penalty, maximum_correction, score, probability


def _new_hierarchical_calibrator(c_value: float) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("score", StandardScaler(), ["LOGIT_SCORE"]),
            (
                "groups",
                OneHotEncoder(handle_unknown="ignore"),
                list(CALIBRATION_CATEGORICAL_COLUMNS),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=3000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _calibration_frame(probability: np.ndarray, strata: pd.DataFrame) -> pd.DataFrame:
    frame = strata[list(CALIBRATION_CATEGORICAL_COLUMNS)].astype(str).reset_index(drop=True)
    frame.insert(0, "LOGIT_SCORE", logit(np.clip(probability, 1e-6, 1 - 1e-6)))
    return frame


def _select_hierarchical_calibrator(
    probability: np.ndarray,
    target: np.ndarray,
    strata: pd.DataFrame,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, np.ndarray, Pipeline]:
    features = _calibration_frame(probability, strata)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 700)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for c_value in (0.001, 0.01, 0.1, 1.0):
        calibrated = np.full(len(target), np.nan, dtype=float)
        for fit_indices, score_indices in splitter.split(features, target, groups):
            model = _new_hierarchical_calibrator(c_value)
            model.fit(
                features.iloc[fit_indices],
                target[fit_indices],
                model__sample_weight=weights[fit_indices],
            )
            calibrated[score_indices] = model.predict_proba(features.iloc[score_indices])[:, 1]
        if not np.isfinite(calibrated).all():
            raise ValueError("Hierarchical calibration left rows without predictions")
        score = float(log_loss(target, calibrated, sample_weight=weights, labels=[0, 1]))
        candidates.append((score, c_value, calibrated))
    score, c_value, calibrated = min(candidates, key=lambda item: (item[0], item[1]))
    final_model = _new_hierarchical_calibrator(c_value)
    final_model.fit(features, target, model__sample_weight=weights)
    return c_value, score, calibrated, final_model


def _best_blend_weight(
    baseline: np.ndarray,
    challenger: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    brier_tolerance: float,
) -> float:
    baseline_brier = float(brier_score_loss(target, baseline, sample_weight=weights))
    candidates: list[tuple[float, float]] = []
    for challenger_weight in np.linspace(0.0, 1.0, 21):
        probability = (1.0 - challenger_weight) * baseline + challenger_weight * challenger
        brier = float(brier_score_loss(target, probability, sample_weight=weights))
        if brier <= baseline_brier * (1.0 + brier_tolerance) + 1e-12:
            loss = float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))
            candidates.append((loss, float(challenger_weight)))
    if not candidates:
        return 0.0
    return min(candidates, key=lambda item: (item[0], -item[1]))[1]


def fit_adaptive_blender(
    baseline: np.ndarray,
    challenger: np.ndarray,
    target: np.ndarray,
    strata: pd.DataFrame,
    weights: np.ndarray,
    *,
    minimum_group_n: int = 100,
    shrinkage_n: float = 200.0,
    brier_tolerance: float = 0.05,
) -> tuple[AdaptiveBlender, pd.DataFrame]:
    global_weight = _best_blend_weight(
        baseline, challenger, target, weights, brier_tolerance=brier_tolerance
    )
    group_weights: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for column in CALIBRATION_CATEGORICAL_COLUMNS:
        values = strata[column].astype(str).to_numpy()
        lookup: dict[str, float] = {}
        for value in sorted(np.unique(values)):
            mask = values == value
            count = int(mask.sum())
            local_weight = 0.0
            if count >= minimum_group_n:
                local_weight = _best_blend_weight(
                    baseline[mask],
                    challenger[mask],
                    target[mask],
                    weights[mask],
                    brier_tolerance=brier_tolerance,
                )
            if count >= minimum_group_n:
                reliability = count / (count + shrinkage_n)
                shrunk_weight = reliability * local_weight + (1.0 - reliability) * global_weight
                lookup[value] = float(min(global_weight, shrunk_weight))
            else:
                reliability = 0.0
                lookup[value] = 0.0
            rows.append(
                {
                    "stratification": column,
                    "stratum": value,
                    "n": count,
                    "global_challenger_weight": global_weight,
                    "local_challenger_weight": local_weight,
                    "reliability": reliability,
                    "shrunk_challenger_weight": lookup[value],
                }
            )
        group_weights[column] = lookup
    return AdaptiveBlender(global_weight, group_weights), pd.DataFrame(rows)


def _maximum_stratum_brier_degradation(
    baseline: np.ndarray,
    challenger: np.ndarray,
    target: np.ndarray,
    strata: pd.DataFrame,
    weights: np.ndarray,
    *,
    minimum_stratum_n: int,
) -> float:
    degradations: list[float] = []
    for column in ("SOURCE", "ERA", "REGION"):
        values = strata[column].astype(str).to_numpy()
        for value in np.unique(values):
            mask = values == value
            if int(mask.sum()) < minimum_stratum_n:
                continue
            baseline_brier = float(
                brier_score_loss(target[mask], baseline[mask], sample_weight=weights[mask])
            )
            challenger_brier = float(
                brier_score_loss(target[mask], challenger[mask], sample_weight=weights[mask])
            )
            degradations.append(
                challenger_brier / baseline_brier - 1.0 if baseline_brier > 0 else np.inf
            )
    return max(degradations, default=0.0)


def select_adaptive_blender(
    baseline: np.ndarray,
    challenger: np.ndarray,
    target: np.ndarray,
    strata: pd.DataFrame,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[AdaptiveBlender, pd.DataFrame, float | None, float, float, np.ndarray]:
    """Select shrinkage using cross-fitted outer-training predictions."""

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 800)
    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for shrinkage_n in (25.0, 50.0, 100.0, 200.0, 500.0):
        probability = np.full(len(target), np.nan, dtype=float)
        for fit_indices, score_indices in splitter.split(strata, target, groups):
            blender, _diagnostics = fit_adaptive_blender(
                baseline[fit_indices],
                challenger[fit_indices],
                target[fit_indices],
                strata.iloc[fit_indices],
                weights[fit_indices],
                shrinkage_n=shrinkage_n,
            )
            probability[score_indices], _row_weights = blender.predict(
                baseline[score_indices],
                challenger[score_indices],
                strata.iloc[score_indices],
            )
        if not np.isfinite(probability).all():
            raise ValueError("Adaptive-blend selection left rows without predictions")
        score = float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))
        maximum_degradation = _maximum_stratum_brier_degradation(
            baseline,
            probability,
            target,
            strata,
            weights,
            minimum_stratum_n=50,
        )
        candidates.append((score, maximum_degradation, shrinkage_n, probability))
    eligible = [candidate for candidate in candidates if candidate[1] <= 0.10]
    if not eligible:
        return (
            AdaptiveBlender(global_weight=0.0, group_weights={}),
            pd.DataFrame(),
            None,
            float(log_loss(target, baseline, sample_weight=weights, labels=[0, 1])),
            0.0,
            baseline.copy(),
        )
    score, maximum_degradation, shrinkage_n, probability = min(
        eligible, key=lambda item: (item[0], item[1], item[2])
    )
    final_blender, diagnostics = fit_adaptive_blender(
        baseline,
        challenger,
        target,
        strata,
        weights,
        shrinkage_n=shrinkage_n,
    )
    return (
        final_blender,
        diagnostics,
        shrinkage_n,
        score,
        maximum_degradation,
        probability,
    )


def _fit_intercept_only_calibration(
    probability: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Fit one log-odds offset without changing ranking or calibration slope."""

    score = logit(np.clip(probability, 1e-6, 1 - 1e-6))
    normalized_weights = weights / weights.sum()

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        linear = score + value[0]
        loss = float(np.sum(normalized_weights * (np.logaddexp(0.0, linear) - target * linear)))
        gradient = np.asarray([np.sum(normalized_weights * (expit(linear) - target))], dtype=float)
        return loss, gradient

    fit = minimize(objective, np.zeros(1), method="BFGS", jac=True)
    if not np.isfinite(fit.x[0]):
        raise ValueError("Intercept-only calibration produced a non-finite offset")
    return float(fit.x[0])


def _strata(oof: pd.DataFrame, seascape: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(oof["SIGHTING_DATE"])
    result = pd.DataFrame(index=oof.index)
    result["SOURCE"] = oof["SOURCE"].fillna("UNKNOWN").astype(str)
    result["ERA"] = pd.cut(
        dates.dt.year,
        bins=[1979, 1999, 2009, 2019, np.inf],
        labels=["1980-1999", "2000-2009", "2010-2019", "2020+"],
    ).astype(str)
    result["REGION"] = [
        h3.latlng_to_cell(lat, lon, 4)
        for lat, lon in zip(oof["LATITUDE"], oof["LONGITUDE"], strict=True)
    ]
    result["EVIDENCE_REGIME"] = oof["EVIDENCE_REGIME"].fillna("UNKNOWN").astype(str)
    availability_columns = [column for column in seascape if column.endswith("__cell_available")]
    availability = seascape[availability_columns].mean(axis=1)
    result["SEASCAPE_COVERAGE"] = np.select(
        [availability.ge(0.999), availability.gt(0.0)],
        ["COMPLETE", "PARTIAL"],
        default="ABSENT",
    )
    return result


def _offset_coefficients(
    model: OffsetResidualModel,
    columns: Sequence[str],
    *,
    fold: int,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(model.coefficient))[::-1]
    return [
        {
            "outer_fold": fold,
            "feature": columns[index],
            "coefficient": float(model.coefficient[index]),
            "absolute_coefficient": float(abs(model.coefficient[index])),
        }
        for index in order[:50]
        if abs(model.coefficient[index]) > 1e-12
    ]


def run_guarded_residual_experiment(
    output_dir: Path,
    paths: ReleasePaths | None = None,
) -> dict[str, Any]:
    paths = paths or resolve_release_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    imputer = SelectiveDateContextImputer.load(paths.model_dir / "ecotype_imputer.joblib")
    oof = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    oof["MODEL_CLASS"] = oof["MODEL_CLASS"].str.upper()
    if oof["ENCOUNTER_ID"].duplicated().any():
        raise ValueError("Guarded residual evaluation requires one row per encounter")

    reference = _encounter_reference(paths)
    binary_reference = reference.loc[reference["ECOTYPE_DETAIL"].isin(["SRKW", "TRANSIENT"])].copy()
    binary_reference["MODEL_CLASS"] = binary_reference["ECOTYPE_DETAIL"]
    weights = _poststratification_weights(oof, binary_reference, label_column="MODEL_CLASS")
    target = oof["Y_TRUE"].to_numpy(int)
    groups = oof["ENCOUNTER_ID"].astype(str).to_numpy()
    kernels = transport_kernel_grid()
    anchors = imputer.anchors_.copy()
    seasonal = build_seasonal_features(oof)
    seascape_root = paths.repo_root / "data/processed/domain/environmental_layer/seascape"
    seascape, seascape_coverage, seascape_lineage = load_static_seascape_features(
        oof, seascape_root
    )
    strata = _strata(oof, seascape)
    for column in CALIBRATION_CATEGORICAL_COLUMNS:
        oof[column] = strata[column]

    prediction_names = (
        "constrained_current_transport_blend",
        "bounded_offset_residual_raw",
        "global_calibrated_residual",
        "hierarchical_calibrated_residual",
        "adaptive_guarded_global_residual",
        "adaptive_guarded_hierarchical_residual",
        "intercept_calibrated_adaptive_hierarchical",
    )
    predictions = {name: np.full(len(oof), np.nan) for name in prediction_names}
    global_challenger_weights = np.full(len(oof), np.nan)
    hierarchical_challenger_weights = np.full(len(oof), np.nan)
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    adaptive_rows: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    outer = StratifiedGroupKFold(
        n_splits=int(imputer.config.model.n_splits),
        shuffle=True,
        random_state=int(imputer.config.model.random_state),
    )

    for fold, (train_indices, test_indices) in enumerate(outer.split(oof, target, groups), start=1):
        test_encounters = set(oof.iloc[test_indices]["ENCOUNTER_ID"].astype(str))
        allowed_anchors = anchors.loc[
            ~anchors["ENCOUNTER_ID"].astype(str).isin(test_encounters)
        ].copy()
        transport_support, support_diagnostics = build_transport_support(
            oof,
            allowed_anchors,
            marine_lookup=imputer.feature_builder.marine_lookup,
            kernels=kernels,
        )
        support_rows.append({"outer_fold": fold, **support_diagnostics})
        train_transport_design = _transport_design(
            transport_support.iloc[train_indices], kernels, directions=("same", "past", "future")
        )
        test_transport_design = _transport_design(
            transport_support.iloc[test_indices], kernels, directions=("same", "past", "future")
        )
        transport_c, transport_loss, transport_train_oof = _select_sparse_logistic(
            train_transport_design,
            target[train_indices],
            groups[train_indices],
            weights[train_indices],
        )
        transport_model = _new_sparse_logistic(transport_c)
        transport_model.fit(
            train_transport_design,
            target[train_indices],
            model__sample_weight=weights[train_indices],
        )
        transport_test = transport_model.predict_proba(test_transport_design)[:, 1]
        current_train = oof.iloc[train_indices]["P_SRKW"].to_numpy(float)
        current_test = oof.iloc[test_indices]["P_SRKW"].to_numpy(float)
        baseline_current_weight, baseline_loss, baseline_degradation = _select_constrained_blend(
            current_train,
            transport_train_oof,
            target[train_indices],
            oof.iloc[train_indices]["SOURCE"].astype(str).to_numpy(),
            weights[train_indices],
        )
        baseline_train = (
            baseline_current_weight * current_train
            + (1.0 - baseline_current_weight) * transport_train_oof
        )
        baseline_test = (
            baseline_current_weight * current_test
            + (1.0 - baseline_current_weight) * transport_test
        )
        predictions["constrained_current_transport_blend"][test_indices] = baseline_test

        train_design = build_reduced_residual_design(
            baseline_train,
            transport_train_oof,
            transport_support.iloc[train_indices],
            seasonal.iloc[train_indices],
            seascape.iloc[train_indices],
        )
        test_design = build_reduced_residual_design(
            baseline_test,
            transport_test,
            transport_support.iloc[test_indices],
            seasonal.iloc[test_indices],
            seascape.iloc[test_indices],
        )
        l2_penalty, maximum_correction, offset_loss, residual_train_oof = _select_offset_residual(
            train_design,
            target[train_indices],
            baseline_train,
            groups[train_indices],
            weights[train_indices],
        )
        residual_model = _fit_offset_residual(
            train_design,
            target[train_indices],
            baseline_train,
            weights[train_indices],
            l2_penalty=l2_penalty,
        )
        residual_test, test_correction = residual_model.predict_probability(
            test_design, baseline_test, maximum_correction=maximum_correction
        )
        predictions["bounded_offset_residual_raw"][test_indices] = residual_test
        coefficient_rows.extend(
            _offset_coefficients(residual_model, train_design.columns, fold=fold)
        )

        global_train_oof, global_calibrator = _crossfit_platt_calibration(
            residual_train_oof,
            target[train_indices],
            groups[train_indices],
            weights[train_indices],
        )
        global_test = global_calibrator.predict_proba(
            logit(np.clip(residual_test, 1e-6, 1 - 1e-6))[:, None]
        )[:, 1]
        predictions["global_calibrated_residual"][test_indices] = global_test

        calibration_c, calibration_loss, hierarchical_train_oof, calibrator = (
            _select_hierarchical_calibrator(
                residual_train_oof,
                target[train_indices],
                strata.iloc[train_indices],
                groups[train_indices],
                weights[train_indices],
            )
        )
        hierarchical_test = calibrator.predict_proba(
            _calibration_frame(residual_test, strata.iloc[test_indices])
        )[:, 1]
        predictions["hierarchical_calibrated_residual"][test_indices] = hierarchical_test

        (
            global_blender,
            global_diagnostics,
            global_shrinkage_n,
            global_blend_loss,
            global_degradation,
            _global_adaptive_train_oof,
        ) = select_adaptive_blender(
            baseline_train,
            global_train_oof,
            target[train_indices],
            strata.iloc[train_indices],
            groups[train_indices],
            weights[train_indices],
        )
        global_adaptive_test, fold_global_weights = global_blender.predict(
            baseline_test, global_test, strata.iloc[test_indices]
        )
        predictions["adaptive_guarded_global_residual"][test_indices] = global_adaptive_test
        global_challenger_weights[test_indices] = fold_global_weights
        if not global_diagnostics.empty:
            global_diagnostics.insert(0, "variant", "global")
            global_diagnostics.insert(0, "outer_fold", fold)
            adaptive_rows.append(global_diagnostics)

        (
            hierarchical_blender,
            hierarchical_diagnostics,
            hierarchical_shrinkage_n,
            hierarchical_blend_loss,
            hierarchical_degradation,
            hierarchical_adaptive_train_oof,
        ) = select_adaptive_blender(
            baseline_train,
            hierarchical_train_oof,
            target[train_indices],
            strata.iloc[train_indices],
            groups[train_indices],
            weights[train_indices],
        )
        hierarchical_adaptive_test, fold_hierarchical_weights = hierarchical_blender.predict(
            baseline_test, hierarchical_test, strata.iloc[test_indices]
        )
        predictions["adaptive_guarded_hierarchical_residual"][
            test_indices
        ] = hierarchical_adaptive_test
        hierarchical_challenger_weights[test_indices] = fold_hierarchical_weights
        active_train = np.abs(hierarchical_adaptive_train_oof - baseline_train) > 1e-12
        active_test = fold_hierarchical_weights > 0.0
        intercept_adjustment = (
            _fit_intercept_only_calibration(
                hierarchical_adaptive_train_oof[active_train],
                target[train_indices][active_train],
                weights[train_indices][active_train],
            )
            if active_train.any()
            else 0.0
        )
        intercept_calibrated_test = baseline_test.copy()
        intercept_calibrated_test[active_test] = expit(
            logit(np.clip(hierarchical_adaptive_test[active_test], 1e-6, 1 - 1e-6))
            + intercept_adjustment
        )
        predictions["intercept_calibrated_adaptive_hierarchical"][
            test_indices
        ] = intercept_calibrated_test
        if not hierarchical_diagnostics.empty:
            hierarchical_diagnostics.insert(0, "variant", "hierarchical")
            hierarchical_diagnostics.insert(0, "outer_fold", fold)
            adaptive_rows.append(hierarchical_diagnostics)
        fold_rows.append(
            {
                "outer_fold": fold,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "transport_selected_c": transport_c,
                "transport_inner_log_loss": transport_loss,
                "baseline_current_weight": baseline_current_weight,
                "baseline_inner_log_loss": baseline_loss,
                "baseline_max_source_brier_degradation": baseline_degradation,
                "residual_feature_n": train_design.shape[1],
                "residual_l2_penalty": l2_penalty,
                "residual_maximum_logit_correction": maximum_correction,
                "residual_inner_log_loss": offset_loss,
                "residual_optimizer_converged": residual_model.converged,
                "test_correction_abs_mean": float(np.abs(test_correction).mean()),
                "test_correction_abs_max": float(np.abs(test_correction).max()),
                "hierarchical_calibration_c": calibration_c,
                "hierarchical_calibration_inner_log_loss": calibration_loss,
                "global_adaptive_shrinkage_n": global_shrinkage_n,
                "global_adaptive_inner_log_loss": global_blend_loss,
                "global_adaptive_max_stratum_brier_degradation": global_degradation,
                "global_adaptive_challenger_weight": global_blender.global_weight,
                "global_adaptive_test_weight_mean": float(fold_global_weights.mean()),
                "hierarchical_adaptive_shrinkage_n": hierarchical_shrinkage_n,
                "hierarchical_adaptive_inner_log_loss": hierarchical_blend_loss,
                "hierarchical_adaptive_max_stratum_brier_degradation": (hierarchical_degradation),
                "hierarchical_adaptive_challenger_weight": (hierarchical_blender.global_weight),
                "hierarchical_adaptive_test_weight_mean": float(fold_hierarchical_weights.mean()),
                "hierarchical_adaptive_test_weight_min": float(fold_hierarchical_weights.min()),
                "hierarchical_adaptive_test_weight_max": float(fold_hierarchical_weights.max()),
                "hierarchical_active_calibration_n": int(active_train.sum()),
                "hierarchical_intercept_adjustment": intercept_adjustment,
            }
        )

    oof["current_model"] = oof["P_SRKW"].to_numpy(float)
    for name, probability in predictions.items():
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"{name} produced invalid probabilities")
        oof[name] = probability
    if (
        not np.isfinite(global_challenger_weights).all()
        or not np.isfinite(hierarchical_challenger_weights).all()
    ):
        raise ValueError("Adaptive blending left rows without challenger weights")
    oof["GLOBAL_ADAPTIVE_CHALLENGER_WEIGHT"] = global_challenger_weights
    oof["HIERARCHICAL_ADAPTIVE_CHALLENGER_WEIGHT"] = hierarchical_challenger_weights

    model_names = ("current_model", *prediction_names)
    comparison = pd.DataFrame(
        [{"model": name, **binary_metrics(target, oof[name], weights)} for name in model_names]
    )
    stratified = _stratified_metrics(oof, model_names, weights)
    candidate_names = tuple(name for name in prediction_names if name != prediction_names[0])
    gates, gate_summary = _gate_table(
        stratified,
        candidate_names,
        baseline_model="constrained_current_transport_blend",
    )
    pass_by_model = gate_summary.groupby("model")["all_required_strata_pass"].all()
    baseline_metrics = comparison.set_index("model").loc["constrained_current_transport_blend"]
    eligible: list[str] = []
    for model_name in candidate_names:
        metrics = comparison.set_index("model").loc[model_name]
        if (
            bool(pass_by_model.get(model_name, False))
            and float(metrics["brier"]) < float(baseline_metrics["brier"])
            and float(metrics["log_loss"]) < float(baseline_metrics["log_loss"])
            and float(metrics["equal_mass_ece_10"]) <= 0.03
            and abs(float(metrics["calibration_intercept"])) <= 0.10
            and 0.8 <= float(metrics["calibration_slope"]) <= 1.2
        ):
            eligible.append(model_name)
    eligible_comparison = comparison.loc[comparison["model"].isin(eligible)]
    recommended = (
        str(eligible_comparison.sort_values(["log_loss", "brier"]).iloc[0]["model"])
        if not eligible_comparison.empty
        else "constrained_current_transport_blend"
    )
    bootstrap = {
        name: _paired_bootstrap(
            target,
            oof["constrained_current_transport_blend"].to_numpy(float),
            oof[name].to_numpy(float),
            weights,
        )
        for name in candidate_names
    }

    prediction_export = oof[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "SOURCE",
            "MODEL_CLASS",
            "Y_TRUE",
            *CALIBRATION_CATEGORICAL_COLUMNS,
            *model_names,
            "GLOBAL_ADAPTIVE_CHALLENGER_WEIGHT",
            "HIERARCHICAL_ADAPTIVE_CHALLENGER_WEIGHT",
        ]
    ].copy()
    frames = {
        "guarded_residual_comparison": comparison,
        "guarded_residual_metrics_by_stratum": stratified,
        "guarded_residual_stratum_gates": gates,
        "guarded_residual_gate_summary": gate_summary,
        "guarded_residual_fold_selections": pd.DataFrame(fold_rows),
        "guarded_residual_top_coefficients": pd.DataFrame(coefficient_rows),
        "guarded_residual_adaptive_weights": (
            pd.concat(adaptive_rows, ignore_index=True) if adaptive_rows else pd.DataFrame()
        ),
        "guarded_residual_seascape_coverage": seascape_coverage,
        "guarded_residual_transport_support": pd.DataFrame(support_rows),
        "guarded_residual_oof_predictions": prediction_export,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    seascape_manifest = seascape_root / "seascape_release_manifest.json"
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "experiment": "guarded offset residual with regularized hierarchical calibration",
            "release_id": paths.release_id,
            "snapshot_id": paths.snapshot_id,
            "release_manifest": paths.manifest_path,
            "model_sha256": imputer.training_summary_.get("MODEL_IDENTITY_SHA256"),
            "water_graph_sha256": _water_graph_sha256(imputer.feature_builder.marine_lookup.graph),
            "seascape_release_manifest_sha256": checksum_path(seascape_manifest),
            "seascape_artifacts": seascape_lineage,
            "reduced_seascape_features": list(REDUCED_SEASCAPE_FEATURES),
            "seasonal_features": list(seasonal.columns),
            "residual_semantics": "bounded additive correction to safe transport baseline log odds",
            "source_used_as_biological_predictor": False,
            "source_usage": "regularized calibration and adaptive reliability shrinkage only",
            "adaptive_blend_groups": list(CALIBRATION_CATEGORICAL_COLUMNS),
            "gate_baseline": "constrained_current_transport_blend",
            "promotion_metrics": {
                "ece_maximum": 0.03,
                "calibration_intercept_absolute_maximum": 0.10,
                "calibration_slope_range": [0.8, 1.2],
                "maximum_required_stratum_brier_degradation": 0.10,
            },
            "bootstrap_vs_safe_transport": bootstrap,
            "recommended_research_challenger": recommended,
            "promotion_eligible": False,
            "promotion_blockers": [
                "binary target remains conditional on SRKW-versus-Transient membership",
                "P_OTHER and abstention remain separate and are not produced by this binary experiment",
                "only three random encounter outer folds are available",
                "rolling-origin, spatial-blocked, and leave-one-source-out confirmation remains required",
                "no blinded unknown-label audit sample exists",
            ],
            "outputs": [f"{name}.csv" for name in frames],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        **frames,
        "bootstrap": bootstrap,
        "recommended_research_challenger": recommended,
        "manifest_path": manifest_path,
    }


__all__ = [
    "AdaptiveBlender",
    "REDUCED_SEASCAPE_FEATURES",
    "build_reduced_residual_design",
    "fit_adaptive_blender",
    "run_guarded_residual_experiment",
]
