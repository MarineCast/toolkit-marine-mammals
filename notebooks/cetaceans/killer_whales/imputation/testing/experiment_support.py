"""Reproducible, non-production experiments for sightings ecotype imputation.

The helpers in this module read one immutable sightings release and write only
under the adjacent ``outputs`` directory.  They deliberately do not update a
model pointer, canonical Parquet path, or release manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import resolve_sightings_release_artifact

RANDOM_SEED = 20260820
BINARY_LABELS = {"TRANSIENT": 0, "SRKW": 1}
KNOWN_OTHER_LABELS = {"NRKW", "OFFSHORE", "OTHER"}
THREE_CLASS_ORDER = ("SRKW", "TRANSIENT", "OTHER")
EVALUATION_STRATEGIES = ("reconstruction", "encounter", "blocked", "purged_blocked")


@dataclass(frozen=True)
class ReleasePaths:
    repo_root: Path
    manifest_path: Path
    release_id: str
    snapshot_id: str
    observations: Path
    diagnostics: Path
    model_dir: Path


def find_repo_root(start: Path | None = None) -> Path:
    """Resolve an explicitly selected data workspace, never discover a repository."""
    import os
    selected = start if start is not None else os.environ.get("MARINE_MAMMALS_WORKSPACE_ROOT")
    if selected is None:
        raise ValueError("Pass repo_root=DATA_WORKSPACE or set MARINE_MAMMALS_WORKSPACE_ROOT")
    root = Path(selected).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def resolve_release_paths(
    repo_root: Path | None = None,
    *,
    release_manifest: Path | None = None,
) -> ReleasePaths:
    root = find_repo_root(repo_root)
    if release_manifest is None:
        release_reference = (
            root / "data/processed/domain/whale_layer/sightings/releases/latest.json"
        )
    else:
        release_reference = release_manifest.expanduser().resolve()
    observations = resolve_sightings_release_artifact(
        release_reference, "whale.sightings.observations"
    )
    diagnostics = resolve_sightings_release_artifact(
        release_reference, "whale.sightings.imputation_diagnostics_retrospective"
    )
    model = resolve_sightings_release_artifact(
        release_reference, "whale.sightings.imputation_model"
    )
    return ReleasePaths(
        repo_root=root,
        manifest_path=observations.manifest_path,
        release_id=observations.release_id,
        snapshot_id=str(observations.snapshot_id),
        observations=observations.path,
        diagnostics=diagnostics.path,
        model_dir=model.path,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def current_model_metrics(paths: ReleasePaths) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    source_frames: list[pd.DataFrame] = []
    for strategy in EVALUATION_STRATEGIES:
        strategy_root = paths.model_dir / "metrics" / strategy
        summary = json.loads((strategy_root / "summary.json").read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "strategy": strategy,
                "n": int(summary["N"]),
                "brier": float(summary["BRIER"]),
                "log_loss": float(summary["LOG_LOSS"]),
                "roc_auc": float(summary["ROC_AUC"]),
                "ece_10": float(summary["ECE_10"]),
                "coverage": float(summary["COVERAGE"]),
                "accepted_n": int(summary["ACCEPTED_N"]),
                "selective_error": float(summary["SELECTIVE_ERROR"]),
                "selective_error_upper": float(summary["SELECTIVE_ERROR_UPPER"]),
                "forced_accuracy": float(summary["FORCED_ACCURACY"]),
            }
        )
        source = pd.read_csv(strategy_root / "metrics_by_source_era_region.csv")
        source = source.loc[source["STRATIFICATION"].eq("SOURCE")].copy()
        source.insert(0, "strategy", strategy)
        source_frames.append(source)
    model_metrics = json.loads((paths.model_dir / "model_metrics.json").read_text())
    training = model_metrics["training_summary"]
    metadata = {
        "release_id": paths.release_id,
        "snapshot_id": paths.snapshot_id,
        "release_manifest": paths.manifest_path,
        "model_dir": paths.model_dir,
        "model_sha256": model_metrics["model_sha256"],
        "model_version": training["MODEL_VERSION"],
        "fit_run_id": training["FIT_RUN_ID"],
        "labeled_n": training["LABELED_N"],
        "srkw_n": training["SRKW_N"],
        "transient_n": training["TRANSIENT_N"],
        "known_other_anchors_n": training["KNOWN_OTHER_ANCHORS_N"],
        "unknown_query_n": training["UNKNOWN_QUERY_N"],
        "mixed_ecotype_encounter_n": training["MIXED_ECOTYPE_ENCOUNTER_N"],
        "soft_count_certified": bool(training["SOFT_COUNT_CERTIFICATION"]["CERTIFIED"]),
        "hard_class_certification": training["CLASS_CERTIFICATION"],
        "open_set_status": training["OPEN_SET_STATUS"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return pd.DataFrame(summary_rows), pd.concat(source_frames, ignore_index=True), metadata


def write_current_model_outputs(
    output_dir: Path,
    paths: ReleasePaths,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, by_source, metadata = current_model_metrics(paths)
    encounter_oof = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    recomputed_metrics = binary_metrics(encounter_oof["Y_TRUE"], encounter_oof["P_SRKW"])
    accepted = encounter_oof["ACCEPTED"].astype(bool)
    accepted_error = (
        float(
            encounter_oof.loc[accepted, "PREDICTED_CLASS"]
            .ne(encounter_oof.loc[accepted, "MODEL_CLASS"])
            .mean()
        )
        if accepted.any()
        else float("nan")
    )
    reconciliation = pd.DataFrame(
        [
            {
                "metric": "brier",
                "persisted": summary.loc[summary["strategy"].eq("encounter"), "brier"].iat[0],
                "recomputed": recomputed_metrics["brier"],
            },
            {
                "metric": "log_loss",
                "persisted": summary.loc[summary["strategy"].eq("encounter"), "log_loss"].iat[0],
                "recomputed": recomputed_metrics["log_loss"],
            },
            {
                "metric": "roc_auc",
                "persisted": summary.loc[summary["strategy"].eq("encounter"), "roc_auc"].iat[0],
                "recomputed": recomputed_metrics["roc_auc"],
            },
            {
                "metric": "coverage",
                "persisted": summary.loc[summary["strategy"].eq("encounter"), "coverage"].iat[0],
                "recomputed": float(accepted.mean()),
            },
            {
                "metric": "selective_error",
                "persisted": summary.loc[
                    summary["strategy"].eq("encounter"), "selective_error"
                ].iat[0],
                "recomputed": accepted_error,
            },
        ]
    )
    if not np.allclose(
        reconciliation["persisted"],
        reconciliation["recomputed"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("Persisted encounter metrics do not reconcile with OOF predictions")
    summary_path = output_dir / "current_model_summary.csv"
    source_path = output_dir / "current_model_metrics_by_source.csv"
    reconciliation_path = output_dir / "encounter_metric_reconciliation.csv"
    summary.to_csv(summary_path, index=False)
    by_source.to_csv(source_path, index=False)
    reconciliation.to_csv(reconciliation_path, index=False)
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            **metadata,
            "experiment": "current pooled binary imputation baseline",
            "promotion_eligible": False,
            "evaluation_note": (
                "Metrics reproduce the immutable fitted model outputs. The evaluation "
                "sample is capped and class-balanced and is not a deployment-prevalence estimate."
            ),
            "outputs": [summary_path.name, source_path.name, reconciliation_path.name],
        },
    )
    return {
        "summary": summary,
        "by_source": by_source,
        "reconciliation": reconciliation,
        "metadata": metadata,
        "manifest_path": manifest_path,
    }


def _equal_mass_ece(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    order = np.argsort(probability, kind="mergesort")
    total_weight = float(weights.sum())
    result = 0.0
    for indices in np.array_split(order, bins):
        if len(indices) == 0:
            continue
        bin_weight = float(weights[indices].sum())
        if bin_weight <= 0:
            continue
        observed = float(np.average(y_true[indices], weights=weights[indices]))
        predicted = float(np.average(probability[indices], weights=weights[indices]))
        result += (bin_weight / total_weight) * abs(observed - predicted)
    return result


def _calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    score = logit(np.clip(probability, 1e-6, 1 - 1e-6))

    def objective(parameters: np.ndarray) -> float:
        fitted = expit(parameters[0] + parameters[1] * score)
        return float(log_loss(y_true, fitted, sample_weight=weights, labels=[0, 1]))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return float(result.x[0]), float(result.x[1])


def binary_metrics(
    y_true: Iterable[int],
    probability: Iterable[float],
    weights: Iterable[float] | None = None,
) -> dict[str, float | int]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(probability), dtype=float), 1e-9, 1 - 1e-9)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(list(weights), dtype=float)
    intercept, slope = _calibration_intercept_slope(y, p, w)
    if np.unique(y).size >= 2:
        auc = float(roc_auc_score(y, p, sample_weight=w))
        average_precision = float(average_precision_score(y, p, sample_weight=w))
    else:
        auc = float("nan")
        average_precision = float(np.average(y, weights=w))
    return {
        "n": int(len(y)),
        "effective_weight": float(w.sum()),
        "prevalence": float(np.average(y, weights=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "roc_auc": auc,
        "average_precision": average_precision,
        "equal_mass_ece_10": _equal_mass_ece(y, p, w),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _encounter_reference(paths: ReleasePaths) -> pd.DataFrame:
    observation_columns = [
        "OBSERVATION_ID",
        "SIGHTING_DATE",
        "LATITUDE",
        "LONGITUDE",
        "ECOTYPE_DETAIL",
        "SOURCE",
        "SOURCE_REPORT_COUNT",
        "COORDINATE_UNCERTAINTY_M",
        "OBSERVATION_QUALITY_TIER",
        "SOURCE_TIME_PRECISION",
    ]
    observations = pd.read_parquet(paths.observations, columns=observation_columns)
    diagnostics = pd.read_parquet(
        paths.diagnostics,
        columns=["OBSERVATION_ID", "ENCOUNTER_ID", "ENCOUNTER_LABEL_CONFLICT"],
    )
    frame = observations.merge(diagnostics, on="OBSERVATION_ID", validate="one_to_one")
    frame = frame.loc[~frame["ENCOUNTER_LABEL_CONFLICT"].fillna(False)].copy()
    frame["ECOTYPE_DETAIL"] = frame["ECOTYPE_DETAIL"].fillna("UNKNOWN").str.upper()
    family = np.select(
        [
            frame["ECOTYPE_DETAIL"].isin(BINARY_LABELS),
            frame["ECOTYPE_DETAIL"].isin(KNOWN_OTHER_LABELS),
        ],
        ["MODELED", "OTHER"],
        default="EXCLUDED",
    )
    frame["TARGET_FAMILY"] = family
    contradictory = (
        frame.loc[frame["TARGET_FAMILY"].ne("EXCLUDED")]
        .groupby("ENCOUNTER_ID")["TARGET_FAMILY"]
        .nunique()
    )
    bad_encounters = set(contradictory[contradictory.gt(1)].index)
    frame = frame.loc[~frame["ENCOUNTER_ID"].isin(bad_encounters)].copy()
    quality_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    frame["_QUALITY_RANK"] = (
        frame["OBSERVATION_QUALITY_TIER"].fillna("").str.upper().map(quality_rank).fillna(0)
    )
    return (
        frame.sort_values(
            ["ENCOUNTER_ID", "_QUALITY_RANK", "SOURCE_REPORT_COUNT", "OBSERVATION_ID"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates("ENCOUNTER_ID", keep="first")
        .drop(columns="_QUALITY_RANK")
        .reset_index(drop=True)
    )


def _poststratification_weights(
    evaluation: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    label_column: str,
    strata: Sequence[str] = ("SOURCE",),
) -> np.ndarray:
    keys = [*strata, label_column]
    target = reference.groupby(keys, dropna=False).size().astype(float)
    target /= float(target.sum())
    sample = evaluation.groupby(keys, dropna=False).size().astype(float)
    sample /= float(sample.sum())
    weights: list[float] = []
    for row in evaluation[keys].itertuples(index=False, name=None):
        sample_share = float(sample.get(row, 0.0))
        target_share = float(target.get(row, 0.0))
        weights.append(target_share / sample_share if sample_share > 0 else 0.0)
    result = np.asarray(weights, dtype=float)
    if not np.isfinite(result).all() or result.sum() <= 0:
        raise ValueError("Post-stratification weights are invalid")
    return result / float(result.mean())


def _make_calibrator(categories: Sequence[str], *, c_value: float) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            ["LOGIT_RAW"],
        )
    ]
    if categories:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categories),
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(C=c_value, max_iter=2000, random_state=RANDOM_SEED),
            ),
        ]
    )


def _fit_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
) -> Pipeline:
    pipeline.fit(features, target, model__sample_weight=weights)
    return pipeline


def nested_source_aware_calibration(
    oof: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    features = oof[["SOURCE", "ERA", "EVIDENCE_REGIME"]].copy()
    features["LOGIT_RAW"] = logit(np.clip(oof["P_SRKW_RAW"].to_numpy(float), 1e-6, 1 - 1e-6))
    y = oof["Y_TRUE"].to_numpy(int)
    groups = oof["ENCOUNTER_ID"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.full(len(oof), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    candidates = [("pooled_natural_prior", (), c_value) for c_value in (0.05, 0.2, 1.0)] + [
        ("source_era_regime", ("SOURCE", "ERA", "EVIDENCE_REGIME"), c_value)
        for c_value in (0.05, 0.2, 1.0)
    ]
    for outer_fold, (train, test) in enumerate(outer.split(features, y, groups), start=1):
        inner = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_SEED + outer_fold,
        )
        candidate_scores: list[tuple[float, str, Sequence[str], float]] = []
        for name, categories, c_value in candidates:
            losses: list[float] = []
            inner_features = features.iloc[train]
            inner_y = y[train]
            inner_groups = groups[train]
            for inner_train, inner_test in inner.split(inner_features, inner_y, inner_groups):
                fit_indices = train[inner_train]
                score_indices = train[inner_test]
                model = _fit_pipeline(
                    _make_calibrator(categories, c_value=c_value),
                    features.iloc[fit_indices],
                    y[fit_indices],
                    weights[fit_indices],
                )
                probability = model.predict_proba(features.iloc[score_indices])[:, 1]
                losses.append(
                    float(
                        log_loss(
                            y[score_indices],
                            probability,
                            sample_weight=weights[score_indices],
                            labels=[0, 1],
                        )
                    )
                )
            candidate_scores.append((float(np.mean(losses)), name, categories, c_value))
        score, name, categories, c_value = min(
            candidate_scores,
            key=lambda item: (item[0], item[1], item[3]),
        )
        final_model = _fit_pipeline(
            _make_calibrator(categories, c_value=c_value),
            features.iloc[train],
            y[train],
            weights[train],
        )
        predictions[test] = final_model.predict_proba(features.iloc[test])[:, 1]
        selections.append(
            {
                "outer_fold": outer_fold,
                "selected_model": name,
                "selected_c": c_value,
                "inner_weighted_log_loss": score,
                "train_n": len(train),
                "test_n": len(test),
            }
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Nested calibration did not produce one prediction per row")
    return predictions, pd.DataFrame(selections)


def _source_metric_table(
    frame: pd.DataFrame,
    probability_columns: Sequence[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, indices in frame.groupby("SOURCE", sort=True).groups.items():
        positions = frame.index.get_indexer(indices)
        for column in probability_columns:
            rows.append(
                {
                    "source": source,
                    "model": column,
                    **binary_metrics(
                        frame.loc[indices, "Y_TRUE"],
                        frame.loc[indices, column],
                        weights[positions],
                    ),
                }
            )
    return pd.DataFrame(rows)


def leave_one_source_out_calibration(
    oof: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {"LOGIT_RAW": logit(np.clip(oof["P_SRKW_RAW"].to_numpy(float), 1e-6, 1 - 1e-6))}
    )
    y = oof["Y_TRUE"].to_numpy(int)
    rows: list[dict[str, Any]] = []
    for source in sorted(oof["SOURCE"].dropna().unique()):
        test = oof["SOURCE"].eq(source).to_numpy()
        train = ~test
        model = _fit_pipeline(
            _make_calibrator((), c_value=0.2),
            features.loc[train],
            y[train],
            weights[train],
        )
        probability = model.predict_proba(features.loc[test])[:, 1]
        current = binary_metrics(y[test], oof.loc[test, "P_SRKW"], weights[test])
        challenger = binary_metrics(y[test], probability, weights[test])
        rows.append(
            {
                "source": source,
                "n": int(test.sum()),
                "current_brier": current["brier"],
                "challenger_brier": challenger["brier"],
                "current_log_loss": current["log_loss"],
                "challenger_log_loss": challenger["log_loss"],
                "challenger_ece": challenger["equal_mass_ece_10"],
            }
        )
    return pd.DataFrame(rows)


def _feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    observed = pd.to_datetime(result["SIGHTING_DATE"], errors="coerce")
    month = observed.dt.month.fillna(1).astype(float)
    result["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    result["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)
    uncertainty = pd.to_numeric(result["COORDINATE_UNCERTAINTY_M"], errors="coerce")
    result["LOG_COORDINATE_UNCERTAINTY"] = np.log1p(uncertainty.clip(lower=0))
    return result


def _make_open_set_model(*, c_value: float) -> Pipeline:
    numeric = [
        "LATITUDE",
        "LONGITUDE",
        "MONTH_SIN",
        "MONTH_COS",
        "LOG_COORDINATE_UNCERTAINTY",
        "SOURCE_REPORT_COUNT",
    ]
    categorical = ["OBSERVATION_QUALITY_TIER", "SOURCE_TIME_PRECISION"]
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                        (
                            "quality",
                            Pipeline(
                                [
                                    (
                                        "impute",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "encode",
                                        OneHotEncoder(handle_unknown="ignore"),
                                    ),
                                ]
                            ),
                            categorical,
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def nested_open_set_probabilities(open_frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    features = _feature_engineering(open_frame)
    y = open_frame["OPEN_TARGET"].to_numpy(int)
    groups = open_frame["ENCOUNTER_ID"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 100)
    predictions = np.full(len(open_frame), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    for outer_fold, (train, test) in enumerate(outer.split(features, y, groups), start=1):
        inner = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_SEED + 100 + outer_fold,
        )
        candidates: list[tuple[float, float]] = []
        for c_value in (0.02, 0.1, 0.5, 2.0):
            losses: list[float] = []
            inner_features = features.iloc[train]
            inner_y = y[train]
            inner_groups = groups[train]
            for inner_train, inner_test in inner.split(inner_features, inner_y, inner_groups):
                fit_indices = train[inner_train]
                score_indices = train[inner_test]
                model = _make_open_set_model(c_value=c_value)
                model.fit(features.iloc[fit_indices], y[fit_indices])
                probability = model.predict_proba(features.iloc[score_indices])[:, 1]
                losses.append(float(log_loss(y[score_indices], probability, labels=[0, 1])))
            candidates.append((float(np.mean(losses)), c_value))
        score, c_value = min(candidates, key=lambda item: (item[0], item[1]))
        model = _make_open_set_model(c_value=c_value)
        model.fit(features.iloc[train], y[train])
        predictions[test] = model.predict_proba(features.iloc[test])[:, 1]
        selections.append(
            {
                "outer_fold": outer_fold,
                "selected_c": c_value,
                "inner_log_loss": score,
                "train_other_n": int((y[train] == 0).sum()),
                "test_other_n": int((y[test] == 0).sum()),
            }
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Open-set cross-fitting did not produce one prediction per row")
    return predictions, pd.DataFrame(selections)


def _crossfit_open_threshold(
    open_frame: pd.DataFrame,
    probability_modeled: np.ndarray,
    *,
    target_other_recall: float = 0.90,
) -> tuple[np.ndarray, pd.DataFrame]:
    y = open_frame["OPEN_TARGET"].to_numpy(int)
    groups = open_frame["ENCOUNTER_ID"].astype(str).to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 200)
    predicted_other = np.zeros(len(open_frame), dtype=bool)
    rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(folds.split(open_frame, y, groups), start=1):
        other_probability = probability_modeled[train][y[train] == 0]
        threshold = float(np.quantile(other_probability, target_other_recall, method="higher"))
        predicted_other[test] = probability_modeled[test] <= threshold
        rows.append(
            {
                "fold": fold,
                "p_modeled_threshold": threshold,
                "train_other_n": int((y[train] == 0).sum()),
                "test_other_n": int((y[test] == 0).sum()),
            }
        )
    return predicted_other, pd.DataFrame(rows)


def _multiclass_metrics(
    labels: Sequence[str],
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    encoded = np.asarray([THREE_CLASS_ORDER.index(label) for label in labels], dtype=int)
    one_hot = np.eye(len(THREE_CLASS_ORDER))[encoded]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return {
        "n": int(len(encoded)),
        "multiclass_brier": float(
            np.average(np.square(clipped - one_hot).sum(axis=1), weights=weights)
        ),
        "multiclass_log_loss": float(
            log_loss(
                encoded,
                clipped,
                sample_weight=weights,
                labels=list(range(len(THREE_CLASS_ORDER))),
            )
        ),
    }


def _paired_bootstrap(
    y: np.ndarray,
    current: np.ndarray,
    challenger: np.ndarray,
    weights: np.ndarray,
    *,
    replicates: int = 300,
) -> dict[str, float]:
    random = np.random.default_rng(RANDOM_SEED)
    brier_delta: list[float] = []
    log_delta: list[float] = []
    for _ in range(replicates):
        indices = random.integers(0, len(y), size=len(y))
        selected_y = y[indices]
        selected_weights = weights[indices]
        baseline = current[indices]
        improved = challenger[indices]
        brier_delta.append(
            float(
                brier_score_loss(selected_y, baseline, sample_weight=selected_weights)
                - brier_score_loss(selected_y, improved, sample_weight=selected_weights)
            )
        )
        log_delta.append(
            float(
                log_loss(selected_y, baseline, sample_weight=selected_weights, labels=[0, 1])
                - log_loss(selected_y, improved, sample_weight=selected_weights, labels=[0, 1])
            )
        )
    return {
        "brier_improvement_median": float(np.median(brier_delta)),
        "brier_improvement_ci_low": float(np.quantile(brier_delta, 0.025)),
        "brier_improvement_ci_high": float(np.quantile(brier_delta, 0.975)),
        "log_loss_improvement_median": float(np.median(log_delta)),
        "log_loss_improvement_ci_low": float(np.quantile(log_delta, 0.025)),
        "log_loss_improvement_ci_high": float(np.quantile(log_delta, 0.975)),
    }


def run_improved_experiment(
    output_dir: Path,
    paths: ReleasePaths,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    if oof["ENCOUNTER_ID"].duplicated().any():
        raise ValueError("Encounter evaluation contains duplicate independent units")
    reference = _encounter_reference(paths)
    binary_reference = reference.loc[reference["ECOTYPE_DETAIL"].isin(BINARY_LABELS)].copy()
    binary_reference["MODEL_CLASS"] = binary_reference["ECOTYPE_DETAIL"]
    oof["MODEL_CLASS"] = oof["MODEL_CLASS"].str.upper()
    weights = _poststratification_weights(
        oof,
        binary_reference,
        label_column="MODEL_CLASS",
    )
    improved_probability, calibrator_selection = nested_source_aware_calibration(oof, weights)
    oof["P_SRKW_CURRENT"] = oof["P_SRKW"].to_numpy(float)
    natural_prevalence = float(np.average(oof["Y_TRUE"], weights=weights))
    oof["P_SRKW_PREVALENCE"] = natural_prevalence
    local_total = oof["LOCAL_SRKW_SUPPORT"] + oof["LOCAL_TRANSIENT_SUPPORT"]
    local_prior_strength = 0.25
    oof["P_SRKW_LOCAL_SUPPORT"] = (
        oof["LOCAL_SRKW_SUPPORT"] + local_prior_strength * natural_prevalence
    ) / (local_total + local_prior_strength)
    oof["P_SRKW_SOURCE_AWARE"] = improved_probability

    binary_comparison = pd.DataFrame(
        [
            {
                "model": "natural_prevalence_constant",
                **binary_metrics(oof["Y_TRUE"], oof["P_SRKW_PREVALENCE"], weights),
            },
            {
                "model": "transparent_local_support",
                **binary_metrics(oof["Y_TRUE"], oof["P_SRKW_LOCAL_SUPPORT"], weights),
            },
            {
                "model": "current_balanced_calibration",
                **binary_metrics(oof["Y_TRUE"], oof["P_SRKW_CURRENT"], weights),
            },
            {
                "model": "nested_source_era_natural_prior",
                **binary_metrics(oof["Y_TRUE"], improved_probability, weights),
            },
        ]
    )
    source_metrics = _source_metric_table(
        oof,
        [
            "P_SRKW_LOCAL_SUPPORT",
            "P_SRKW_CURRENT",
            "P_SRKW_SOURCE_AWARE",
        ],
        weights,
    )
    source_transport = leave_one_source_out_calibration(oof, weights)
    bootstrap = _paired_bootstrap(
        oof["Y_TRUE"].to_numpy(int),
        oof["P_SRKW_CURRENT"].to_numpy(float),
        improved_probability,
        weights,
    )

    feature_columns = [
        "OBSERVATION_ID",
        "SIGHTING_DATE",
        "LATITUDE",
        "LONGITUDE",
        "SOURCE",
        "SOURCE_REPORT_COUNT",
        "COORDINATE_UNCERTAINTY_M",
        "OBSERVATION_QUALITY_TIER",
        "SOURCE_TIME_PRECISION",
    ]
    observation_features = pd.read_parquet(paths.observations, columns=feature_columns)
    modeled = oof[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "MODEL_CLASS",
            "P_SRKW_CURRENT",
            "P_SRKW_SOURCE_AWARE",
        ]
    ].merge(observation_features, on="OBSERVATION_ID", validate="one_to_one")
    modeled["OPEN_TARGET"] = 1
    modeled["THREE_CLASS"] = modeled["MODEL_CLASS"]
    other = reference.loc[reference["ECOTYPE_DETAIL"].isin(KNOWN_OTHER_LABELS)].copy()
    other = other.loc[~other["ENCOUNTER_ID"].isin(set(modeled["ENCOUNTER_ID"]))]
    other["OPEN_TARGET"] = 0
    other["THREE_CLASS"] = "OTHER"
    other["MODEL_CLASS"] = "OTHER"
    other["P_SRKW_CURRENT"] = 0.5
    other["P_SRKW_SOURCE_AWARE"] = 0.5
    open_frame = pd.concat(
        [modeled, other[modeled.columns]],
        ignore_index=True,
        sort=False,
    )
    open_probability, open_selection = nested_open_set_probabilities(open_frame)
    predicted_other, open_thresholds = _crossfit_open_threshold(open_frame, open_probability)
    open_y = open_frame["OPEN_TARGET"].to_numpy(int)
    open_prevalence = float(open_y.mean())
    open_prevalence_probability = np.full(len(open_y), open_prevalence)
    open_metrics = {
        "known_other_n": int((open_y == 0).sum()),
        "modeled_n": int((open_y == 1).sum()),
        "modeled_prevalence": open_prevalence,
        "other_recall": float(recall_score(open_y == 0, predicted_other)),
        "modeled_retention": float(recall_score(open_y == 1, ~predicted_other)),
        "balanced_accuracy": float(balanced_accuracy_score(open_y, (~predicted_other).astype(int))),
        "open_probability_roc_auc": float(roc_auc_score(open_y, open_probability)),
        "open_probability_brier": float(brier_score_loss(open_y, open_probability)),
        "open_probability_log_loss": float(log_loss(open_y, open_probability, labels=[0, 1])),
        "prevalence_baseline_brier": float(brier_score_loss(open_y, open_prevalence_probability)),
        "prevalence_baseline_log_loss": float(
            log_loss(open_y, open_prevalence_probability, labels=[0, 1])
        ),
        "production_certified": False,
        "certification_blocker": (
            "Only the small existing known-Other set is available; no blinded, "
            "double-reviewed unknown-label audit sample exists."
        ),
    }

    three_reference = reference.loc[
        reference["ECOTYPE_DETAIL"].isin(set(BINARY_LABELS) | KNOWN_OTHER_LABELS)
    ].copy()
    three_reference["THREE_CLASS"] = three_reference["ECOTYPE_DETAIL"].where(
        three_reference["ECOTYPE_DETAIL"].isin(BINARY_LABELS), "OTHER"
    )
    open_weights = _poststratification_weights(
        open_frame,
        three_reference,
        label_column="THREE_CLASS",
    )
    natural_class_probability = np.asarray(
        [
            np.average(open_frame["THREE_CLASS"].eq(label), weights=open_weights)
            for label in THREE_CLASS_ORDER
        ]
    )
    current_conditional = open_frame["P_SRKW_CURRENT"].to_numpy(float)
    improved_conditional = open_frame["P_SRKW_SOURCE_AWARE"].to_numpy(float)
    epsilon = 1e-12
    current_probabilities = np.column_stack(
        [
            (1 - epsilon) * current_conditional,
            (1 - epsilon) * (1 - current_conditional),
            np.full(len(open_frame), epsilon),
        ]
    )
    hierarchical_probabilities = np.column_stack(
        [
            open_probability * improved_conditional,
            open_probability * (1 - improved_conditional),
            1 - open_probability,
        ]
    )
    multiclass_comparison = pd.DataFrame(
        [
            {
                "model": "three_class_prevalence",
                **_multiclass_metrics(
                    open_frame["THREE_CLASS"],
                    np.tile(natural_class_probability, (len(open_frame), 1)),
                    open_weights,
                ),
            },
            {
                "model": "current_forced_binary",
                **_multiclass_metrics(
                    open_frame["THREE_CLASS"], current_probabilities, open_weights
                ),
            },
            {
                "model": "hierarchical_open_set_prototype",
                **_multiclass_metrics(
                    open_frame["THREE_CLASS"], hierarchical_probabilities, open_weights
                ),
            },
        ]
    )

    frames = {
        "binary_comparison": binary_comparison,
        "binary_metrics_by_source": source_metrics,
        "leave_one_source_out": source_transport,
        "calibrator_selections": calibrator_selection,
        "open_set_selections": open_selection,
        "open_set_thresholds": open_thresholds,
        "multiclass_comparison": multiclass_comparison,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    write_json(output_dir / "open_set_metrics.json", open_metrics)
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "experiment": "nested source-aware calibration and hierarchical open-set prototype",
            "release_id": paths.release_id,
            "snapshot_id": paths.snapshot_id,
            "release_manifest": paths.manifest_path,
            "input_strategy": "encounter",
            "outer_folds": 5,
            "inner_folds": 3,
            "poststratification": "SOURCE x observed class encounter prevalence",
            "source_usage": "calibration and diagnostics only; not an open-set feature",
            "open_set_features": [
                "latitude",
                "longitude",
                "season",
                "coordinate uncertainty",
                "report count",
                "quality tier",
                "time precision",
            ],
            "bootstrap": bootstrap,
            "open_set_metrics": open_metrics,
            "promotion_eligible": False,
            "promotion_blockers": [
                "no blinded double-reviewed unknown-label audit sample",
                "known-Other sample below production minimum",
                "experiment begins from the current capped encounter OOF sample",
                "rolling-origin and repeated spatial outer folds remain to be run",
            ],
            "outputs": [f"{name}.csv" for name in frames] + ["open_set_metrics.json"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        **frames,
        "open_set_metrics": open_metrics,
        "bootstrap": bootstrap,
        "manifest_path": manifest_path,
    }


__all__ = [
    "ReleasePaths",
    "binary_metrics",
    "current_model_metrics",
    "find_repo_root",
    "resolve_release_paths",
    "run_improved_experiment",
    "write_current_model_outputs",
]
