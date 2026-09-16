"""Research-only model-strategy bake-off for ecotype imputation.

The experiment starts from the immutable encounter-held-out scores produced by
the current imputer and the notebook-local marine transport experiment.  Every
new learner is cross-fitted again at the encounter level.  Source is never a
biological feature; it is used only to define robustness groups and diagnostics.

This is deliberately not a production trainer.  In particular, the rolling,
spatial, and leave-one-source-out exercises operate over frozen OOF base scores;
they are useful residual-model stress tests, but are not substitutes for an
end-to-end refit of the underlying imputer and anchor graph in each split.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import h3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, logsumexp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from marine_mammal_toolkit.tools._core.checksums import checksum_path

from covariate_experiment import build_seasonal_features, load_static_seascape_features
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
from guarded_residual_experiment import REDUCED_SEASCAPE_FEATURES


BASELINE_MODEL = "safe_transport_baseline"
DIFFUSION_PROBABILITY_COLUMNS = (
    "current_model",
    "fixed_kernel_direct",
    "learned_single_kernel",
    "learned_multiscale_past_only",
    "learned_multiscale_retrospective",
    "constrained_current_transport_blend",
)
CANDIDATE_MODELS = (
    "regularized_logistic",
    "spline_gam_residual",
    "group_robust_hist_boost",
    "learned_graph_diffusion_stack",
    "group_dro_logistic",
    "evidence_regime_mixture",
)
ENSEMBLE_MODEL = "calibrated_constrained_ensemble"
PRIMARY_SCHEME = "encounter_5fold"
STRESS_SCHEMES = (
    "rolling_origin",
    "spatial_leave_region_out",
    "leave_one_source_out",
)


@dataclass
class StrategyDataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    weights: np.ndarray
    seascape_coverage: pd.DataFrame
    seascape_lineage: list[dict[str, Any]]
    input_paths: dict[str, Path]


@dataclass
class PipelineProbabilityModel:
    pipeline: Pipeline

    def predict(self, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        del metadata
        return self.pipeline.predict_proba(features)[:, 1]


@dataclass
class GroupDROModel:
    preprocessor: Pipeline
    classifier: LogisticRegression
    group_weights: dict[str, float]

    def predict(self, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        del metadata
        transformed = self.preprocessor.transform(features)
        return self.classifier.predict_proba(transformed)[:, 1]


@dataclass
class GraphDiffusionStack:
    weights: np.ndarray
    columns: tuple[str, ...]

    def predict(self, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        del metadata
        matrix = np.clip(features.loc[:, self.columns].to_numpy(float), 1e-6, 1 - 1e-6)
        return np.clip(matrix @ self.weights, 1e-6, 1 - 1e-6)


@dataclass
class EvidenceMixtureModel:
    pooled: Pipeline
    experts: dict[str, Pipeline]
    expert_weights: dict[str, float]

    def predict(self, features: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
        pooled = self.pooled.predict_proba(features)[:, 1]
        result = pooled.copy()
        regimes = metadata["EVIDENCE_GROUP"].astype(str).to_numpy()
        for regime, expert in self.experts.items():
            selected = regimes == regime
            if not selected.any():
                continue
            expert_probability = expert.predict_proba(features.loc[selected])[:, 1]
            weight = self.expert_weights[regime]
            result[selected] = (1.0 - weight) * pooled[selected] + weight * expert_probability
        return np.clip(result, 1e-6, 1 - 1e-6)


def _testing_root() -> Path:
    return Path(__file__).resolve().parent


def _validate_cached_experiment(
    path: Path,
    *,
    release_id: str,
    snapshot_id: str,
) -> None:
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_id") != release_id:
        raise ValueError(f"Cached experiment release mismatch: {path}")
    if manifest.get("snapshot_id") != snapshot_id:
        raise ValueError(f"Cached experiment snapshot mismatch: {path}")


def _collapse_evidence_regime(values: pd.Series) -> pd.Series:
    text = values.fillna("UNKNOWN").astype(str).str.upper()
    return pd.Series(
        np.select(
            [
                text.eq("SAME_DAY"),
                text.eq("LAGGED"),
                text.isin(["NO_LOCAL_CANDIDATES", "WATER_BLOCKED", "MARINE_LOOKUP_MISSING"]),
                text.str.contains("CONFLICT", regex=False),
            ],
            ["SAME_DAY", "LAGGED", "NO_ROUTED_SUPPORT", "CONFLICT"],
            default="WEAK_OR_OTHER",
        ),
        index=values.index,
    )


def load_strategy_dataset(paths: ReleasePaths | None = None) -> StrategyDataset:
    """Build one immutable feature table shared by every challenger."""

    paths = paths or resolve_release_paths()
    transport_path = _testing_root() / "outputs/transport/transport_oof_predictions.csv"
    if not transport_path.is_file():
        raise FileNotFoundError("Run 03_learned_spatiotemporal_transport.ipynb first")
    _validate_cached_experiment(
        transport_path,
        release_id=paths.release_id,
        snapshot_id=paths.snapshot_id,
    )

    current = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    transport = pd.read_csv(transport_path)
    if current["ENCOUNTER_ID"].duplicated().any():
        raise ValueError("Strategy evaluation requires one representative per encounter")
    transport_columns = ["OBSERVATION_ID", *DIFFUSION_PROBABILITY_COLUMNS]
    if transport["OBSERVATION_ID"].duplicated().any():
        raise ValueError("Transport OOF output contains duplicate observations")
    frame = current.merge(
        transport.loc[:, transport_columns],
        on="OBSERVATION_ID",
        validate="one_to_one",
        suffixes=("", "__transport"),
    )
    if not frame["Y_TRUE"].isin([0, 1]).all():
        raise ValueError("Binary strategy target contains values outside {0, 1}")

    reference = _encounter_reference(paths)
    binary_reference = reference.loc[
        reference["ECOTYPE_DETAIL"].isin(["SRKW", "TRANSIENT"])
    ].copy()
    binary_reference["MODEL_CLASS"] = binary_reference["ECOTYPE_DETAIL"]
    weights = _poststratification_weights(frame, binary_reference, label_column="MODEL_CLASS")

    dates = pd.to_datetime(frame["SIGHTING_DATE"], errors="raise")
    frame["ERA"] = pd.cut(
        dates.dt.year,
        bins=[1979, 1999, 2009, 2019, np.inf],
        labels=["1980-1999", "2000-2009", "2010-2019", "2020+"],
    ).astype(str)
    frame["REGION"] = [
        h3.latlng_to_cell(latitude, longitude, 4)
        for latitude, longitude in zip(frame["LATITUDE"], frame["LONGITUDE"], strict=True)
    ]
    frame["EVIDENCE_GROUP"] = _collapse_evidence_regime(frame["EVIDENCE_REGIME"])
    robust_group = frame["SOURCE"].astype(str) + "::" + frame["ERA"].astype(str)
    group_counts = robust_group.value_counts()
    frame["ROBUST_GROUP"] = np.where(
        robust_group.map(group_counts).ge(50),
        robust_group,
        frame["SOURCE"].astype(str) + "::OTHER_ERA",
    )

    seasonal = build_seasonal_features(frame)
    day = dates.dt.dayofyear.to_numpy(float)
    days_in_year = np.where(dates.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    seasonal["season__day_fraction"] = (day - 1.0) / days_in_year
    seascape_root = paths.repo_root / "data/processed/domain/environmental_layer/seascape"
    seascape, seascape_coverage, seascape_lineage = load_static_seascape_features(
        frame, seascape_root
    )

    probability_features: dict[str, np.ndarray] = {}
    for column in DIFFUSION_PROBABILITY_COLUMNS:
        values = np.clip(frame[column].to_numpy(float), 1e-6, 1 - 1e-6)
        probability_features[column] = values
        probability_features[f"logit__{column}"] = logit(values)

    support_columns = [
        "P_SRKW_RAW",
        "OOD_MARGIN",
        "SAME_DAY_OTHER_SUPPORT",
        "SAME_DAY_SRKW_SUPPORT",
        "SAME_DAY_TRANSIENT_SUPPORT",
        "LOCAL_SRKW_SUPPORT",
        "LOCAL_TRANSIENT_SUPPORT",
        "LOCAL_OTHER_SUPPORT",
        "PAST_SRKW_SUPPORT",
        "PAST_TRANSIENT_SUPPORT",
        "FUTURE_SRKW_SUPPORT",
        "FUTURE_TRANSIENT_SUPPORT",
        "NEAREST_SRKW_KM",
        "NEAREST_TRANSIENT_KM",
        "NEAREST_SRKW_DAY_LAG",
        "NEAREST_TRANSIENT_DAY_LAG",
        "LOCAL_BINARY_SUPPORT_MARGIN",
        "LOCAL_BINARY_SUPPORT_LOG_RATIO",
        "MIN_SRKW_IMPLIED_SPEED_KM_DAY",
        "MIN_TRANSIENT_IMPLIED_SPEED_KM_DAY",
        "SRKW_MOVEMENT_CONTINUITY",
        "TRANSIENT_MOVEMENT_CONTINUITY",
        "MARINE_CANDIDATE_NEIGHBORS",
        "MARINE_ROUTED_NEIGHBORS",
        "MARINE_BARRIER_EXCLUDED_NEIGHBORS",
        "MARINE_OUT_OF_RANGE_NEIGHBORS",
        "MARINE_LOOKUP_COVERAGE",
        "MARINE_TARGET_SNAP_KM",
    ]
    missing_support = sorted(set(support_columns) - set(frame.columns))
    if missing_support:
        raise ValueError(f"Current OOF output is missing support fields: {missing_support}")

    selected_seascape = list(REDUCED_SEASCAPE_FEATURES)
    for column in tuple(selected_seascape):
        missing_column = f"{column}__missing"
        if missing_column in seascape:
            selected_seascape.append(missing_column)
    availability_columns = [
        column for column in seascape.columns if column.endswith("__cell_available")
    ]
    selected_seascape.extend(availability_columns)
    design = pd.concat(
        [
            pd.DataFrame(probability_features, index=frame.index),
            frame.loc[:, ["LATITUDE", "LONGITUDE", *support_columns]].apply(
                pd.to_numeric, errors="coerce"
            ),
            seasonal,
            seascape.loc[:, selected_seascape],
        ],
        axis=1,
    )
    design = design.replace([np.inf, -np.inf], np.nan)
    if design.columns.duplicated().any():
        duplicates = design.columns[design.columns.duplicated()].tolist()
        raise ValueError(f"Strategy design has duplicate fields: {duplicates}")
    for column in DIFFUSION_PROBABILITY_COLUMNS:
        if not design[column].between(0, 1).all():
            raise ValueError(f"{column} is not a valid probability")
    input_paths = {
        "current_oof": paths.model_dir / "metrics/encounter/oof_predictions.csv",
        "transport_oof": transport_path,
    }
    return StrategyDataset(
        frame=frame.reset_index(drop=True),
        features=design.reset_index(drop=True),
        weights=np.asarray(weights, dtype=float),
        seascape_coverage=seascape_coverage,
        seascape_lineage=seascape_lineage,
        input_paths=input_paths,
    )


def _fit_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
) -> Pipeline:
    pipeline.fit(features, target, model__sample_weight=weights)
    return pipeline


def _numeric_logistic(*, c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "prepare",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(C=c_value, max_iter=2000, random_state=seed),
            ),
        ]
    )


def _spline_gam(*, seed: int, columns: Sequence[str]) -> Pipeline:
    smooth_candidates = [
        "season__day_fraction",
        "LATITUDE",
        "LONGITUDE",
        "seascape__bathymetry__bathymetry_median",
        "seascape__bathymetry__distance_to_isobath_200_m",
        "seascape__shoreline_proximity__water_network_distance_m",
        "seascape__shoreline_proximity__open_ocean_index",
        "seascape__waterbody_morphometry__local_waterbody_width_m",
        "seascape__estuarine_connectivity__water_network_distance_to_estuary_m",
        "seascape__fluvial_connectivity__water_network_distance_to_fluvial_mouth_m",
    ]
    smooth = [column for column in smooth_candidates if column in columns]
    periodic = ["season__day_fraction"]
    nonperiodic = [column for column in smooth if column not in periodic]
    evidence = [
        column
        for column in columns
        if column.startswith("logit__")
        or "SUPPORT" in column.upper()
        or "MOVEMENT" in column.upper()
        or column.startswith("MARINE_")
    ]
    transformer = ColumnTransformer(
        [
            (
                "periodic",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=8,
                                degree=3,
                                extrapolation="periodic",
                                include_bias=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                periodic,
            ),
            (
                "smooth",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=4,
                                degree=2,
                                extrapolation="linear",
                                include_bias=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                nonperiodic,
            ),
            (
                "evidence",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                evidence,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("prepare", transformer),
            (
                "model",
                LogisticRegression(C=0.05, max_iter=3000, random_state=seed),
            ),
        ]
    )


def group_robust_sample_weights(
    base_weights: Iterable[float],
    groups: Iterable[str],
    *,
    strength: float = 0.75,
) -> np.ndarray:
    """Blend natural weights toward equal total weight per robustness group."""

    weights = np.asarray(list(base_weights), dtype=float)
    group_values = np.asarray(list(groups), dtype=str)
    unique, inverse = np.unique(group_values, return_inverse=True)
    totals = np.bincount(inverse, weights=weights, minlength=len(unique))
    positive = totals[totals > 0]
    target = float(np.exp(np.mean(np.log(positive))))
    multiplier = np.power(target / np.maximum(totals, 1e-12), strength)
    multiplier = np.clip(multiplier, 0.25, 4.0)
    result = weights * multiplier[inverse]
    return result / float(result.mean())


def _fit_group_dro(
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int = 8,
    eta: float = 0.35,
) -> GroupDROModel:
    preprocessor = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    transformed = preprocessor.fit_transform(features)
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    group_mass = np.bincount(inverse, weights=weights, minlength=len(unique))
    q = np.full(len(unique), 1.0 / len(unique), dtype=float)
    classifier: LogisticRegression | None = None
    for iteration in range(iterations):
        multiplier = q[inverse] / np.maximum(group_mass[inverse], 1e-12)
        fit_weights = weights * multiplier
        fit_weights *= len(fit_weights) / fit_weights.sum()
        classifier = LogisticRegression(
            C=0.05,
            max_iter=2000,
            random_state=seed + iteration,
        )
        classifier.fit(transformed, target, sample_weight=fit_weights)
        probability = np.clip(classifier.predict_proba(transformed)[:, 1], 1e-6, 1 - 1e-6)
        losses = -(target * np.log(probability) + (1 - target) * np.log(1 - probability))
        group_losses = np.asarray(
            [
                np.average(losses[inverse == index], weights=weights[inverse == index])
                for index in range(len(unique))
            ]
        )
        log_q = np.log(np.clip(q, 1e-12, None)) + eta * group_losses
        q = np.exp(log_q - logsumexp(log_q))
        q = np.maximum(q, 0.01 / len(unique))
        q /= q.sum()
    if classifier is None:
        raise RuntimeError("Group-DRO did not fit a classifier")
    return GroupDROModel(
        preprocessor=preprocessor,
        classifier=classifier,
        group_weights={name: float(q[index]) for index, name in enumerate(unique)},
    )


def fit_graph_diffusion_stack(
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    source: np.ndarray,
    *,
    robust_penalty: float = 0.05,
) -> GraphDiffusionStack:
    """Fit non-negative simplex weights across water-graph diffusion scales."""

    matrix = np.clip(
        features.loc[:, DIFFUSION_PROBABILITY_COLUMNS].to_numpy(float),
        1e-6,
        1 - 1e-6,
    )
    normalized = weights / weights.sum()
    source_values = source.astype(str)
    unique_sources = np.unique(source_values)

    def objective(simplex: np.ndarray) -> float:
        probability = np.clip(matrix @ simplex, 1e-6, 1 - 1e-6)
        row_loss = -(target * np.log(probability) + (1 - target) * np.log(1 - probability))
        average_loss = float(np.sum(normalized * row_loss))
        source_loss = np.asarray(
            [
                np.average(row_loss[source_values == value], weights=weights[source_values == value])
                for value in unique_sources
            ]
        )
        robust_excess = float(
            0.1 * logsumexp(source_loss / 0.1) - np.average(source_loss)
        )
        return average_loss + robust_penalty * robust_excess

    initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Graph diffusion stack optimization failed: {result.message}")
    fitted = np.clip(result.x, 0.0, 1.0)
    fitted /= fitted.sum()
    return GraphDiffusionStack(fitted, DIFFUSION_PROBABILITY_COLUMNS)


def _fit_evidence_mixture(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
) -> EvidenceMixtureModel:
    pooled = _fit_pipeline(_numeric_logistic(c_value=0.05, seed=seed), features, target, weights)
    experts: dict[str, Pipeline] = {}
    expert_weights: dict[str, float] = {}
    regimes = metadata["EVIDENCE_GROUP"].astype(str).to_numpy()
    for offset, regime in enumerate(sorted(np.unique(regimes)), start=1):
        selected = regimes == regime
        class_counts = np.bincount(target[selected], minlength=2)
        if selected.sum() < 100 or class_counts.min() < 20:
            continue
        expert = _fit_pipeline(
            _numeric_logistic(c_value=0.05, seed=seed + offset),
            features.loc[selected],
            target[selected],
            weights[selected],
        )
        experts[regime] = expert
        effective_minor = float(class_counts.min())
        expert_weights[regime] = min(0.85, effective_minor / (effective_minor + 75.0))
    return EvidenceMixtureModel(pooled, experts, expert_weights)


def _fit_candidate_models(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    columns = list(features.columns)
    regularized = _fit_pipeline(
        _numeric_logistic(c_value=0.05, seed=seed), features, target, weights
    )
    gam = _fit_pipeline(_spline_gam(seed=seed + 1, columns=columns), features, target, weights)
    robust_weights = group_robust_sample_weights(weights, metadata["ROBUST_GROUP"])
    hist = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=120,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=5.0,
                    random_state=seed + 2,
                ),
            ),
        ]
    )
    hist.fit(features, target, model__sample_weight=robust_weights)
    graph = fit_graph_diffusion_stack(
        features,
        target,
        weights,
        metadata["SOURCE"].astype(str).to_numpy(),
    )
    group_dro = _fit_group_dro(
        features,
        target,
        weights,
        metadata["ROBUST_GROUP"].astype(str).to_numpy(),
        seed=seed + 3,
    )
    mixture = _fit_evidence_mixture(
        features, metadata, target, weights, seed=seed + 4
    )
    return {
        "regularized_logistic": PipelineProbabilityModel(regularized),
        "spline_gam_residual": PipelineProbabilityModel(gam),
        "group_robust_hist_boost": PipelineProbabilityModel(hist),
        "learned_graph_diffusion_stack": graph,
        "group_dro_logistic": group_dro,
        "evidence_regime_mixture": mixture,
    }


def _predict_candidates(
    models: dict[str, Any],
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in CANDIDATE_MODELS:
        probability = np.asarray(models[name].predict(features, metadata), dtype=float)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"{name} produced an invalid probability")
        result[name] = np.clip(probability, 1e-6, 1 - 1e-6)
    return result


def _fit_simplex_ensemble(
    probabilities: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    normalized = weights / weights.sum()

    def objective(simplex: np.ndarray) -> float:
        probability = np.clip(probabilities @ simplex, 1e-6, 1 - 1e-6)
        return float(
            np.sum(
                normalized
                * (-(target * np.log(probability) + (1 - target) * np.log(1 - probability)))
            )
        )

    initial = np.full(probabilities.shape[1], 1.0 / probabilities.shape[1])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * probabilities.shape[1],
        constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Ensemble optimization failed: {result.message}")
    result_weights = np.clip(result.x, 0.0, 1.0)
    return result_weights / result_weights.sum()


def _fit_nested_ensemble(
    train_features: pd.DataFrame,
    train_metadata: pd.DataFrame,
    train_target: np.ndarray,
    train_weights: np.ndarray,
    test_probabilities: dict[str, np.ndarray],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensemble_columns = (BASELINE_MODEL, *CANDIDATE_MODELS)
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + 1000)
    inner_probability = np.full((len(train_features), len(ensemble_columns)), np.nan)
    inner_fold = np.full(len(train_features), -1, dtype=int)
    groups = train_metadata["ENCOUNTER_ID"].astype(str).to_numpy()
    for fold, (fit_indices, score_indices) in enumerate(
        inner.split(train_features, train_target, groups)
    ):
        models = _fit_candidate_models(
            train_features.iloc[fit_indices],
            train_metadata.iloc[fit_indices],
            train_target[fit_indices],
            train_weights[fit_indices],
            seed=seed + 100 + fold,
        )
        predicted = _predict_candidates(
            models,
            train_features.iloc[score_indices],
            train_metadata.iloc[score_indices],
        )
        inner_probability[score_indices, 0] = train_features.iloc[score_indices][
            "constrained_current_transport_blend"
        ].to_numpy(float)
        for column_index, name in enumerate(CANDIDATE_MODELS, start=1):
            inner_probability[score_indices, column_index] = predicted[name]
        inner_fold[score_indices] = fold
    if not np.isfinite(inner_probability).all() or (inner_fold < 0).any():
        raise ValueError("Nested ensemble did not receive complete inner OOF predictions")

    selection = inner_fold < 2
    calibration = inner_fold == 2
    ensemble_weights = _fit_simplex_ensemble(
        inner_probability[selection],
        train_target[selection],
        train_weights[selection],
    )
    raw_calibration_probability = np.clip(
        inner_probability[calibration] @ ensemble_weights, 1e-6, 1 - 1e-6
    )
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=seed + 2000)
    calibrator.fit(
        logit(raw_calibration_probability)[:, None],
        train_target[calibration],
        sample_weight=train_weights[calibration],
    )
    calibrated_train = calibrator.predict_proba(logit(raw_calibration_probability)[:, None])[:, 1]
    blend_candidates = np.linspace(0.0, 1.0, 5)
    calibration_losses: list[tuple[float, float]] = []
    for blend in blend_candidates:
        probability = (1.0 - blend) * raw_calibration_probability + blend * calibrated_train
        metrics = binary_metrics(
            train_target[calibration], probability, train_weights[calibration]
        )
        calibration_losses.append((float(metrics["log_loss"]), float(blend)))
    _loss, calibration_blend = min(calibration_losses, key=lambda item: (item[0], item[1]))

    test_matrix = np.column_stack(
        [
            test_probabilities[BASELINE_MODEL],
            *[test_probabilities[name] for name in CANDIDATE_MODELS],
        ]
    )
    raw_test = np.clip(test_matrix @ ensemble_weights, 1e-6, 1 - 1e-6)
    calibrated_test = calibrator.predict_proba(logit(raw_test)[:, None])[:, 1]
    result = (1.0 - calibration_blend) * raw_test + calibration_blend * calibrated_test
    diagnostics = {
        "ensemble_weights": {
            name: float(value) for name, value in zip(ensemble_columns, ensemble_weights, strict=True)
        },
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "calibration_blend": float(calibration_blend),
        "selection_n": int(selection.sum()),
        "calibration_n": int(calibration.sum()),
    }
    return np.clip(result, 1e-6, 1 - 1e-6), diagnostics


def _encounter_splits(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 700)
    target = frame["Y_TRUE"].to_numpy(int)
    groups = frame["ENCOUNTER_ID"].astype(str).to_numpy()
    return [
        (f"encounter_{fold}", train, test)
        for fold, (train, test) in enumerate(splitter.split(frame, target, groups), start=1)
    ]


def _rolling_splits(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(frame["SIGHTING_DATE"])
    windows = (
        (2019, 2020),
        (2021, 2022),
        (2023, 2023),
        (2024, 2024),
        (2025, 2026),
    )
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for start_year, end_year in windows:
        train = np.flatnonzero(dates.dt.year.lt(start_year).to_numpy())
        test = np.flatnonzero(dates.dt.year.between(start_year, end_year).to_numpy())
        if len(train) == 0 or len(test) == 0 or np.unique(frame.iloc[train]["Y_TRUE"]).size < 2:
            continue
        result.append((f"{start_year}_{end_year}", train, test))
    return result


def _spatial_splits(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=5)
    region = frame["REGION"].astype(str).to_numpy()
    return [
        (f"spatial_{fold}", train, test)
        for fold, (train, test) in enumerate(splitter.split(frame, frame["Y_TRUE"], region), start=1)
    ]


def _source_splits(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    source = frame["SOURCE"].astype(str).to_numpy()
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for value in sorted(np.unique(source)):
        test = np.flatnonzero(source == value)
        train = np.flatnonzero(source != value)
        if np.unique(frame.iloc[train]["Y_TRUE"]).size < 2:
            continue
        result.append((f"source_{value}", train, test))
    return result


def _evaluate_scheme(
    dataset: StrategyDataset,
    scheme: str,
    splits: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    include_ensemble: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = dataset.frame
    features = dataset.features
    target = frame["Y_TRUE"].to_numpy(int)
    prediction_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    for fold_number, (fold_name, train, test) in enumerate(splits, start=1):
        models = _fit_candidate_models(
            features.iloc[train],
            frame.iloc[train],
            target[train],
            dataset.weights[train],
            seed=RANDOM_SEED + fold_number * 100,
        )
        predicted = _predict_candidates(models, features.iloc[test], frame.iloc[test])
        predicted[BASELINE_MODEL] = features.iloc[test][
            "constrained_current_transport_blend"
        ].to_numpy(float)
        predicted["current_model"] = features.iloc[test]["current_model"].to_numpy(float)
        if include_ensemble:
            ensemble, diagnostics = _fit_nested_ensemble(
                features.iloc[train].reset_index(drop=True),
                frame.iloc[train].reset_index(drop=True),
                target[train],
                dataset.weights[train],
                predicted,
                seed=RANDOM_SEED + fold_number * 1000,
            )
            predicted[ENSEMBLE_MODEL] = ensemble
            ensemble_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold_name,
                    **diagnostics,
                }
            )
        export = frame.iloc[test][
            [
                "OBSERVATION_ID",
                "ENCOUNTER_ID",
                "SIGHTING_DATE",
                "SOURCE",
                "ERA",
                "REGION",
                "EVIDENCE_GROUP",
                "MODEL_CLASS",
                "Y_TRUE",
            ]
        ].copy()
        export.insert(0, "row_position", test)
        export.insert(0, "fold", fold_name)
        export.insert(0, "scheme", scheme)
        for name, probability in predicted.items():
            export[name] = probability
        export["EVALUATION_WEIGHT"] = dataset.weights[test]
        prediction_rows.append(export)
        fold_rows.append(
            {
                "scheme": scheme,
                "fold": fold_name,
                "train_n": len(train),
                "test_n": len(test),
                "train_start": frame.iloc[train]["SIGHTING_DATE"].min(),
                "train_end": frame.iloc[train]["SIGHTING_DATE"].max(),
                "test_start": frame.iloc[test]["SIGHTING_DATE"].min(),
                "test_end": frame.iloc[test]["SIGHTING_DATE"].max(),
                "train_source_n": frame.iloc[train]["SOURCE"].nunique(),
                "test_source_n": frame.iloc[test]["SOURCE"].nunique(),
                "train_region_n": frame.iloc[train]["REGION"].nunique(),
                "test_region_n": frame.iloc[test]["REGION"].nunique(),
            }
        )
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(ensemble_rows),
    )


def _comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    identity = {
        "scheme",
        "fold",
        "row_position",
        "OBSERVATION_ID",
        "ENCOUNTER_ID",
        "SIGHTING_DATE",
        "SOURCE",
        "ERA",
        "REGION",
        "EVIDENCE_GROUP",
        "MODEL_CLASS",
        "Y_TRUE",
        "EVALUATION_WEIGHT",
    }
    model_names = [column for column in predictions.columns if column not in identity]
    rows: list[dict[str, Any]] = []
    for scheme, subset in predictions.groupby("scheme", sort=False):
        for name in model_names:
            if subset[name].isna().all():
                continue
            valid = subset[name].notna().to_numpy()
            rows.append(
                {
                    "scheme": scheme,
                    "model": name,
                    **binary_metrics(
                        subset.loc[valid, "Y_TRUE"],
                        subset.loc[valid, name],
                        subset.loc[valid, "EVALUATION_WEIGHT"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _primary_stratum_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions["scheme"].eq(PRIMARY_SCHEME)].copy()
    model_names = ["current_model", BASELINE_MODEL, *CANDIDATE_MODELS, ENSEMBLE_MODEL]
    rows: list[dict[str, Any]] = []
    for stratification in ("SOURCE", "ERA", "REGION", "EVIDENCE_GROUP"):
        for value, subset in primary.groupby(stratification, sort=True):
            for model_name in model_names:
                if model_name not in subset or subset[model_name].isna().all():
                    continue
                rows.append(
                    {
                        "stratification": stratification,
                        "stratum": str(value),
                        "model": model_name,
                        **binary_metrics(
                            subset["Y_TRUE"], subset[model_name], subset["EVALUATION_WEIGHT"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _stratum_gates(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = metrics.loc[metrics["model"].eq(BASELINE_MODEL)].set_index(
        ["stratification", "stratum"]
    )
    rows: list[dict[str, Any]] = []
    for model in (*CANDIDATE_MODELS, ENSEMBLE_MODEL):
        challenger = metrics.loc[metrics["model"].eq(model)].set_index(
            ["stratification", "stratum"]
        )
        for key in baseline.index.intersection(challenger.index):
            base_row = baseline.loc[key]
            challenger_row = challenger.loc[key]
            base_brier = float(base_row["brier"])
            challenger_brier = float(challenger_row["brier"])
            degradation = challenger_brier / base_brier - 1 if base_brier > 0 else np.inf
            independent_n = int(base_row["n"])
            rows.append(
                {
                    "model": model,
                    "stratification": key[0],
                    "stratum": key[1],
                    "independent_n": independent_n,
                    "baseline_brier": base_brier,
                    "challenger_brier": challenger_brier,
                    "relative_brier_degradation": degradation,
                    "required_gate": independent_n >= 100,
                    "passes_10pct_degradation_gate": (
                        independent_n < 100 or degradation <= 0.10
                    ),
                }
            )
    gates = pd.DataFrame(rows)
    summary = (
        gates.loc[gates["required_gate"]]
        .groupby(["model", "stratification"], as_index=False)
        .agg(
            required_stratum_n=("stratum", "nunique"),
            failed_stratum_n=(
                "passes_10pct_degradation_gate", lambda values: int((~values).sum())
            ),
            maximum_relative_brier_degradation=("relative_brier_degradation", "max"),
        )
    )
    summary["all_required_strata_pass"] = summary["failed_stratum_n"].eq(0)
    return gates, summary


def _stress_gates(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme in STRESS_SCHEMES:
        subset = comparison.loc[comparison["scheme"].eq(scheme)].set_index("model")
        baseline = subset.loc[BASELINE_MODEL]
        for model in CANDIDATE_MODELS:
            challenger = subset.loc[model]
            brier_degradation = float(challenger["brier"] / baseline["brier"] - 1)
            log_degradation = float(challenger["log_loss"] / baseline["log_loss"] - 1)
            rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "baseline_brier": float(baseline["brier"]),
                    "challenger_brier": float(challenger["brier"]),
                    "relative_brier_degradation": brier_degradation,
                    "relative_log_loss_degradation": log_degradation,
                    "passes_10pct_overall_degradation_gate": (
                        brier_degradation <= 0.10 and log_degradation <= 0.10
                    ),
                }
            )
    return pd.DataFrame(rows)


def _recommendation(
    comparison: pd.DataFrame,
    gate_summary: pd.DataFrame,
    stress_gates: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    primary = comparison.loc[comparison["scheme"].eq(PRIMARY_SCHEME)].set_index("model")
    baseline = primary.loc[BASELINE_MODEL]
    rows: list[dict[str, Any]] = []
    for model in CANDIDATE_MODELS:
        metrics = primary.loc[model]
        stratum_pass = bool(
            gate_summary.loc[gate_summary["model"].eq(model), "all_required_strata_pass"].all()
        )
        stress_pass = bool(
            stress_gates.loc[
                stress_gates["model"].eq(model), "passes_10pct_overall_degradation_gate"
            ].all()
        )
        calibration_pass = (
            abs(float(metrics["calibration_intercept"])) <= 0.10
            and 0.8 <= float(metrics["calibration_slope"]) <= 1.2
            and float(metrics["equal_mass_ece_10"]) <= 0.03
        )
        improves = (
            float(metrics["brier"]) < float(baseline["brier"])
            and float(metrics["log_loss"]) < float(baseline["log_loss"])
        )
        rows.append(
            {
                "model": model,
                "improves_primary_brier_and_log_loss": improves,
                "passes_primary_calibration": calibration_pass,
                "passes_required_strata": stratum_pass,
                "passes_residual_stress_schemes": stress_pass,
                "research_gate_pass": improves and calibration_pass and stratum_pass and stress_pass,
            }
        )
    decisions = pd.DataFrame(rows)
    eligible = decisions.loc[decisions["research_gate_pass"], "model"]
    if eligible.empty:
        return BASELINE_MODEL, decisions
    ranked = primary.loc[list(eligible)].sort_values(["log_loss", "brier"])
    return str(ranked.index[0]), decisions


def run_strategy_bakeoff_experiment(
    output_dir: Path,
    paths: ReleasePaths | None = None,
) -> dict[str, Any]:
    paths = paths or resolve_release_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_strategy_dataset(paths)
    split_factories = {
        PRIMARY_SCHEME: _encounter_splits(dataset.frame),
        "rolling_origin": _rolling_splits(dataset.frame),
        "spatial_leave_region_out": _spatial_splits(dataset.frame),
        "leave_one_source_out": _source_splits(dataset.frame),
    }
    prediction_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    ensemble_parts: list[pd.DataFrame] = []
    for scheme, splits in split_factories.items():
        predictions, folds, ensemble = _evaluate_scheme(
            dataset,
            scheme,
            splits,
            include_ensemble=scheme == PRIMARY_SCHEME,
        )
        prediction_parts.append(predictions)
        fold_parts.append(folds)
        if not ensemble.empty:
            ensemble_parts.append(ensemble)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_diagnostics = pd.concat(fold_parts, ignore_index=True)
    ensemble_diagnostics = (
        pd.concat(ensemble_parts, ignore_index=True) if ensemble_parts else pd.DataFrame()
    )
    comparison = _comparison(predictions)
    stratum_metrics = _primary_stratum_metrics(predictions)
    stratum_gates, gate_summary = _stratum_gates(stratum_metrics)
    stress_gates = _stress_gates(comparison)
    recommendation, research_decisions = _recommendation(
        comparison, gate_summary, stress_gates
    )

    primary = predictions.loc[predictions["scheme"].eq(PRIMARY_SCHEME)].copy()
    bootstrap = {
        model: _paired_bootstrap(
            primary["Y_TRUE"].to_numpy(int),
            primary[BASELINE_MODEL].to_numpy(float),
            primary[model].to_numpy(float),
            primary["EVALUATION_WEIGHT"].to_numpy(float),
        )
        for model in (*CANDIDATE_MODELS, ENSEMBLE_MODEL)
    }

    outputs = {
        "strategy_comparison.csv": comparison,
        "strategy_primary_stratum_metrics.csv": stratum_metrics,
        "strategy_primary_stratum_gates.csv": stratum_gates,
        "strategy_primary_gate_summary.csv": gate_summary,
        "strategy_stress_gates.csv": stress_gates,
        "strategy_research_decisions.csv": research_decisions,
        "strategy_fold_diagnostics.csv": fold_diagnostics,
        "strategy_oof_predictions.csv": predictions,
        "strategy_seascape_coverage.csv": dataset.seascape_coverage,
    }
    if not ensemble_diagnostics.empty:
        serializable = ensemble_diagnostics.copy()
        serializable["ensemble_weights"] = serializable["ensemble_weights"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
        outputs["strategy_ensemble_diagnostics.csv"] = serializable
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)

    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "experiment": "ecotype conditional model strategy bake-off",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_id": paths.release_id,
            "snapshot_id": paths.snapshot_id,
            "release_manifest": paths.manifest_path,
            "input_hashes": {
                name: checksum_path(path) for name, path in dataset.input_paths.items()
            },
            "seascape_artifacts": dataset.seascape_lineage,
            "source_used_as_biological_predictor": False,
            "source_usage": "robustness weighting, group-DRO groups, and diagnostics only",
            "strategies": {
                "regularized_logistic": "regularized linear probability challenger",
                "spline_gam_residual": "periodic season and smooth spatial/seascape effects",
                "group_robust_hist_boost": "histogram boosting with source-era balanced weights",
                "learned_graph_diffusion_stack": (
                    "non-negative robust stack of leakage-controlled water-graph kernels"
                ),
                "group_dro_logistic": "worst-group optimized logistic model without source features",
                "evidence_regime_mixture": "pooled model plus reliability-shrunk regime experts",
                "calibrated_constrained_ensemble": (
                    "nested simplex ensemble with disjoint calibration fold"
                ),
            },
            "evaluation_schemes": list(split_factories),
            "primary_outer_folds": 5,
            "ensemble_inner_folds": 3,
            "recommended_research_challenger": recommendation,
            "bootstrap_vs_safe_transport": bootstrap,
            "promotion_eligible": False,
            "promotion_blockers": [
                "all inputs begin from a capped balanced encounter OOF sample",
                "stress splits reuse frozen base scores rather than refitting the full imputer",
                "rolling-origin is retrospective because future evidence is present in base scores",
                "leave-one-source-out does not refit the underlying imputer or transport anchors",
                "binary target remains conditional on modeled SRKW-versus-Transient membership",
                "no blinded double-reviewed unknown-label audit sample exists",
            ],
            "outputs": list(outputs),
        },
    )
    return {
        "comparison": comparison,
        "stratum_metrics": stratum_metrics,
        "stratum_gates": stratum_gates,
        "gate_summary": gate_summary,
        "stress_gates": stress_gates,
        "research_decisions": research_decisions,
        "fold_diagnostics": fold_diagnostics,
        "predictions": predictions,
        "ensemble_diagnostics": ensemble_diagnostics,
        "recommendation": recommendation,
        "manifest_path": manifest_path,
    }


__all__ = [
    "BASELINE_MODEL",
    "CANDIDATE_MODELS",
    "ENSEMBLE_MODEL",
    "GraphDiffusionStack",
    "StrategyDataset",
    "fit_graph_diffusion_stack",
    "group_robust_sample_weights",
    "load_strategy_dataset",
    "run_strategy_bakeoff_experiment",
]
