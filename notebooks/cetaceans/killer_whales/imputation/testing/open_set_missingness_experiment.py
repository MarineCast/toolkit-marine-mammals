"""Open-set, label-missingness, and active-review experiments.

`P_OTHER` in this module is a biological probability for a labeled non-modeled
ecotype.  It is never used as an abstention flag.  Because the current release
contains only a very small known-Other encounter sample and no blinded audit of
unknown records, every result remains diagnostic and ineligible for soft counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from marine_mammal_toolkit.tools._core.checksums import checksum_path

from covariate_experiment import build_seasonal_features, load_static_seascape_features
from experiment_support import (
    KNOWN_OTHER_LABELS,
    RANDOM_SEED,
    ReleasePaths,
    _crossfit_open_threshold,
    _encounter_reference,
    _multiclass_metrics,
    _poststratification_weights,
    binary_metrics,
    resolve_release_paths,
    write_json,
)
from guarded_residual_experiment import REDUCED_SEASCAPE_FEATURES
from strategy_bakeoff_experiment import (
    _fit_group_dro,
    group_robust_sample_weights,
    load_strategy_dataset,
)


OPEN_MODELS = (
    "open_regularized_logistic",
    "open_spline_gam",
    "open_group_robust_hist_boost",
    "open_group_dro",
    "open_propensity_weighted_hist_boost",
)


@dataclass
class OpenSetDataset:
    reference: pd.DataFrame
    known: pd.DataFrame
    known_features: pd.DataFrame
    reference_features: pd.DataFrame
    open_weights: np.ndarray
    labeled_propensity_oof: np.ndarray
    propensity_metrics: dict[str, Any]
    propensity_by_source: pd.DataFrame
    seascape_coverage: pd.DataFrame
    seascape_lineage: list[dict[str, Any]]


def _era(dates: pd.Series) -> pd.Series:
    years = pd.to_datetime(dates).dt.year
    return pd.cut(
        years,
        bins=[1979, 1999, 2009, 2019, np.inf],
        labels=["1980-1999", "2000-2009", "2010-2019", "2020+"],
    ).astype(str)


def _open_numeric_features(
    frame: pd.DataFrame,
    seascape: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame["SIGHTING_DATE"], errors="raise")
    seasonal = build_seasonal_features(frame)
    day = dates.dt.dayofyear.to_numpy(float)
    days_in_year = np.where(dates.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    seasonal["season__day_fraction"] = (day - 1.0) / days_in_year
    base = pd.DataFrame(
        {
            "LATITUDE": pd.to_numeric(frame["LATITUDE"], errors="coerce"),
            "LONGITUDE": pd.to_numeric(frame["LONGITUDE"], errors="coerce"),
            "log__coordinate_uncertainty_m": np.log1p(
                pd.to_numeric(frame["COORDINATE_UNCERTAINTY_M"], errors="coerce").clip(lower=0)
            ),
            "log__source_report_count": np.log1p(
                pd.to_numeric(frame["SOURCE_REPORT_COUNT"], errors="coerce").clip(lower=0)
            ),
        },
        index=frame.index,
    )
    selected = list(REDUCED_SEASCAPE_FEATURES)
    for column in tuple(selected):
        missing_column = f"{column}__missing"
        if missing_column in seascape:
            selected.append(missing_column)
    selected.extend(
        column for column in seascape.columns if column.endswith("__cell_available")
    )
    result = pd.concat([base, seasonal, seascape[selected]], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def _propensity_features(reference: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(reference["SIGHTING_DATE"], errors="raise")
    month_phase = 2.0 * np.pi * (dates.dt.dayofyear.to_numpy(float) - 1.0) / np.where(
        dates.dt.is_leap_year.to_numpy(), 366.0, 365.0
    )
    return pd.DataFrame(
        {
            "SOURCE": reference["SOURCE"].fillna("UNKNOWN").astype(str),
            "ERA": _era(reference["SIGHTING_DATE"]),
            "OBSERVATION_QUALITY_TIER": reference["OBSERVATION_QUALITY_TIER"]
            .fillna("UNKNOWN")
            .astype(str),
            "SOURCE_TIME_PRECISION": reference["SOURCE_TIME_PRECISION"]
            .fillna("UNKNOWN")
            .astype(str),
            "LATITUDE": pd.to_numeric(reference["LATITUDE"], errors="coerce"),
            "LONGITUDE": pd.to_numeric(reference["LONGITUDE"], errors="coerce"),
            "MONTH_SIN": np.sin(month_phase),
            "MONTH_COS": np.cos(month_phase),
            "LOG_COORDINATE_UNCERTAINTY": np.log1p(
                pd.to_numeric(reference["COORDINATE_UNCERTAINTY_M"], errors="coerce").clip(
                    lower=0
                )
            ),
            "LOG_SOURCE_REPORT_COUNT": np.log1p(
                pd.to_numeric(reference["SOURCE_REPORT_COUNT"], errors="coerce").clip(lower=0)
            ),
        },
        index=reference.index,
    )


def _new_propensity_model(*, seed: int) -> Pipeline:
    numeric = [
        "LATITUDE",
        "LONGITUDE",
        "MONTH_SIN",
        "MONTH_COS",
        "LOG_COORDINATE_UNCERTAINTY",
        "LOG_SOURCE_REPORT_COUNT",
    ]
    categorical = [
        "SOURCE",
        "ERA",
        "OBSERVATION_QUALITY_TIER",
        "SOURCE_TIME_PRECISION",
    ]
    return Pipeline(
        [
            (
                "prepare",
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
                            "categorical",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
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
                LogisticRegression(C=0.2, max_iter=3000, random_state=seed),
            ),
        ]
    )


def crossfit_label_propensity(
    reference: pd.DataFrame,
) -> tuple[np.ndarray, Pipeline, dict[str, Any], pd.DataFrame]:
    features = _propensity_features(reference)
    target = reference["TARGET_FAMILY"].isin(["MODELED", "OTHER"]).astype(int).to_numpy()
    groups = reference["ENCOUNTER_ID"].astype(str).to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 810)
    probability = np.full(len(reference), np.nan)
    for fold, (train, test) in enumerate(folds.split(features, target, groups), start=1):
        model = _new_propensity_model(seed=RANDOM_SEED + 810 + fold)
        model.fit(features.iloc[train], target[train])
        probability[test] = model.predict_proba(features.iloc[test])[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Label-propensity cross-fitting left unscored rows")
    final_model = _new_propensity_model(seed=RANDOM_SEED + 899)
    final_model.fit(features, target)
    metrics = {
        "n": int(len(target)),
        "labeled_n": int(target.sum()),
        "unknown_n": int((target == 0).sum()),
        "labeled_prevalence": float(target.mean()),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability)),
        "minimum_probability": float(probability.min()),
        "maximum_inverse_probability_weight": float(
            (1.0 / np.clip(probability[target == 1], 0.05, 1.0)).max()
        ),
        "identification_warning": (
            "Inverse-propensity weighting assumes labels are missing at random conditional "
            "on these observed fields; this assumption is not testable from current data."
        ),
    }
    diagnostic = reference[["SOURCE", "TARGET_FAMILY"]].copy()
    diagnostic["LABEL_AVAILABLE"] = target.astype(bool)
    diagnostic["P_LABEL_AVAILABLE_OOF"] = probability
    by_source = (
        diagnostic.groupby("SOURCE", as_index=False)
        .agg(
            encounter_n=("LABEL_AVAILABLE", "size"),
            labeled_n=("LABEL_AVAILABLE", "sum"),
            mean_predicted_label_probability=("P_LABEL_AVAILABLE_OOF", "mean"),
            minimum_predicted_label_probability=("P_LABEL_AVAILABLE_OOF", "min"),
        )
    )
    by_source["unknown_n"] = by_source["encounter_n"] - by_source["labeled_n"]
    by_source["observed_label_rate"] = by_source["labeled_n"] / by_source["encounter_n"]
    return probability, final_model, metrics, by_source


def load_open_set_dataset(paths: ReleasePaths | None = None) -> OpenSetDataset:
    paths = paths or resolve_release_paths()
    reference = _encounter_reference(paths)
    reference["ERA"] = _era(reference["SIGHTING_DATE"])
    reference["REGION"] = [
        h3.latlng_to_cell(latitude, longitude, 4)
        for latitude, longitude in zip(
            reference["LATITUDE"], reference["LONGITUDE"], strict=True
        )
    ]
    reference["ROBUST_GROUP"] = (
        reference["SOURCE"].astype(str) + "::" + reference["ERA"].astype(str)
    )
    propensity, _propensity_model, propensity_metrics, propensity_by_source = (
        crossfit_label_propensity(reference)
    )

    current = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    observation_columns = [
        "OBSERVATION_ID",
        "SOURCE_REPORT_COUNT",
        "COORDINATE_UNCERTAINTY_M",
    ]
    observation_features = pd.read_parquet(paths.observations, columns=observation_columns)
    modeled = current[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "LATITUDE",
            "LONGITUDE",
            "SOURCE",
            "OBSERVATION_QUALITY_TIER",
            "SOURCE_TIME_PRECISION",
            "MODEL_CLASS",
            "P_SRKW",
        ]
    ].merge(observation_features, on="OBSERVATION_ID", validate="one_to_one")
    modeled["ECOTYPE_DETAIL"] = modeled["MODEL_CLASS"].str.upper()
    modeled["TARGET_FAMILY"] = "MODELED"
    modeled["ERA"] = _era(modeled["SIGHTING_DATE"])
    modeled["REGION"] = [
        h3.latlng_to_cell(latitude, longitude, 4)
        for latitude, longitude in zip(
            modeled["LATITUDE"], modeled["LONGITUDE"], strict=True
        )
    ]
    modeled["ROBUST_GROUP"] = (
        modeled["SOURCE"].astype(str) + "::" + modeled["ERA"].astype(str)
    )
    modeled["OPEN_TARGET"] = 1
    modeled["THREE_CLASS"] = modeled["MODEL_CLASS"].str.upper()
    other = reference.loc[reference["ECOTYPE_DETAIL"].isin(KNOWN_OTHER_LABELS)].copy()
    other = other.loc[~other["ENCOUNTER_ID"].isin(set(modeled["ENCOUNTER_ID"]))]
    other["OPEN_TARGET"] = 0
    other["THREE_CLASS"] = "OTHER"
    other["MODEL_CLASS"] = "OTHER"
    other["P_SRKW"] = 0.5
    common_columns = [
        "OBSERVATION_ID",
        "ENCOUNTER_ID",
        "SIGHTING_DATE",
        "LATITUDE",
        "LONGITUDE",
        "SOURCE",
        "SOURCE_REPORT_COUNT",
        "COORDINATE_UNCERTAINTY_M",
        "OBSERVATION_QUALITY_TIER",
        "SOURCE_TIME_PRECISION",
        "ECOTYPE_DETAIL",
        "TARGET_FAMILY",
        "ERA",
        "REGION",
        "ROBUST_GROUP",
        "MODEL_CLASS",
        "P_SRKW",
        "OPEN_TARGET",
        "THREE_CLASS",
    ]
    known = pd.concat(
        [modeled[common_columns], other[common_columns]], ignore_index=True, sort=False
    )
    if known["ENCOUNTER_ID"].duplicated().any():
        raise ValueError("Open-set table contains duplicate encounter units")

    seascape_root = paths.repo_root / "data/processed/domain/environmental_layer/seascape"
    reference_seascape, reference_coverage, lineage = load_static_seascape_features(
        reference, seascape_root
    )
    reference_features = _open_numeric_features(reference, reference_seascape)
    known_seascape, known_coverage, known_lineage = load_static_seascape_features(
        known, seascape_root
    )
    if known_lineage != lineage:
        raise ValueError("Static seascape lineage changed between reference and known joins")
    known_features = _open_numeric_features(known, known_seascape)
    reference_coverage.insert(0, "scope", "all_encounter_representatives")
    known_coverage.insert(0, "scope", "known_open_set_evaluation")
    seascape_coverage = pd.concat([reference_coverage, known_coverage], ignore_index=True)

    three_reference = reference.loc[
        reference["ECOTYPE_DETAIL"].isin({"SRKW", "TRANSIENT", *KNOWN_OTHER_LABELS})
    ].copy()
    three_reference["THREE_CLASS"] = three_reference["ECOTYPE_DETAIL"].where(
        three_reference["ECOTYPE_DETAIL"].isin(["SRKW", "TRANSIENT"]), "OTHER"
    )
    open_weights = _poststratification_weights(
        known, three_reference, label_column="THREE_CLASS"
    )
    return OpenSetDataset(
        reference=reference.reset_index(drop=True),
        known=known.reset_index(drop=True),
        known_features=known_features.reset_index(drop=True),
        reference_features=reference_features.reset_index(drop=True),
        open_weights=open_weights,
        labeled_propensity_oof=propensity,
        propensity_metrics=propensity_metrics,
        propensity_by_source=propensity_by_source,
        seascape_coverage=seascape_coverage,
        seascape_lineage=lineage,
    )


def _open_logistic(*, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
            ),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=0.05, max_iter=3000, random_state=seed),
            ),
        ]
    )


def _open_gam(columns: Sequence[str], *, seed: int) -> Pipeline:
    smooth = [
        column
        for column in (
            "LATITUDE",
            "LONGITUDE",
            "seascape__bathymetry__bathymetry_median",
            "seascape__bathymetry__distance_to_isobath_200_m",
            "seascape__shoreline_proximity__water_network_distance_m",
            "seascape__shoreline_proximity__open_ocean_index",
            "seascape__waterbody_morphometry__local_waterbody_width_m",
            "seascape__estuarine_connectivity__water_network_distance_to_estuary_m",
        )
        if column in columns
    ]
    linear = [column for column in columns if column not in {*smooth, "season__day_fraction"}]
    prepare = ColumnTransformer(
        [
            (
                "season",
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
                ["season__day_fraction"],
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
                                include_bias=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                smooth,
            ),
            (
                "linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear,
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", prepare),
            (
                "model",
                LogisticRegression(C=0.03, max_iter=3000, random_state=seed),
            ),
        ]
    )


def _open_hist(*, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=150,
                    max_leaf_nodes=11,
                    min_samples_leaf=30,
                    l2_regularization=8.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def propensity_weights(
    label_probability: Iterable[float],
    *,
    lower_bound: float = 0.05,
) -> np.ndarray:
    probability = np.clip(np.asarray(list(label_probability), dtype=float), lower_bound, 1.0)
    result = 1.0 / probability
    return result / float(result.mean())


def _fit_open_models(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    propensity_weight: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    logistic = _open_logistic(seed=seed)
    logistic.fit(features, target, model__sample_weight=weights)
    gam = _open_gam(features.columns, seed=seed + 1)
    gam.fit(features, target, model__sample_weight=weights)
    robust = group_robust_sample_weights(weights, metadata["ROBUST_GROUP"])
    hist = _open_hist(seed=seed + 2)
    hist.fit(features, target, model__sample_weight=robust)
    group_dro = _fit_group_dro(
        features,
        target,
        weights,
        metadata["ROBUST_GROUP"].astype(str).to_numpy(),
        seed=seed + 3,
        iterations=8,
        eta=0.35,
    )
    propensity_hist = _open_hist(seed=seed + 4)
    propensity_hist.fit(
        features,
        target,
        model__sample_weight=weights * propensity_weight,
    )
    return {
        "open_regularized_logistic": logistic,
        "open_spline_gam": gam,
        "open_group_robust_hist_boost": hist,
        "open_group_dro": group_dro,
        "open_propensity_weighted_hist_boost": propensity_hist,
    }


def _predict_open_models(
    models: dict[str, Any],
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    for name in OPEN_MODELS:
        model = models[name]
        if name == "open_group_dro":
            probability = model.predict(features, metadata)
        else:
            probability = model.predict_proba(features)[:, 1]
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if not np.isfinite(probability).all():
            raise ValueError(f"{name} produced non-finite open-set probabilities")
        probabilities[name] = probability
    return probabilities


def _known_propensity_weights(dataset: OpenSetDataset) -> np.ndarray:
    lookup = pd.Series(
        dataset.labeled_propensity_oof,
        index=dataset.reference["ENCOUNTER_ID"].astype(str),
    )
    probability = lookup.reindex(dataset.known["ENCOUNTER_ID"].astype(str)).to_numpy(float)
    if not np.isfinite(probability).all():
        raise ValueError("Known open-set rows are missing label-propensity estimates")
    return propensity_weights(probability)


def crossfit_open_models(
    dataset: OpenSetDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = dataset.known["OPEN_TARGET"].to_numpy(int)
    groups = dataset.known["ENCOUNTER_ID"].astype(str).to_numpy()
    propensity_weight = _known_propensity_weights(dataset)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 900)
    predictions = {name: np.full(len(target), np.nan) for name in OPEN_MODELS}
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(
        folds.split(dataset.known_features, target, groups), start=1
    ):
        models = _fit_open_models(
            dataset.known_features.iloc[train],
            dataset.known.iloc[train],
            target[train],
            dataset.open_weights[train],
            propensity_weight[train],
            seed=RANDOM_SEED + 900 + fold,
        )
        probability = _predict_open_models(
            models, dataset.known_features.iloc[test], dataset.known.iloc[test]
        )
        for name in OPEN_MODELS:
            predictions[name][test] = probability[name]
        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train),
                "test_n": len(test),
                "train_other_n": int((target[train] == 0).sum()),
                "test_other_n": int((target[test] == 0).sum()),
            }
        )
    export = dataset.known[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "SOURCE",
            "ERA",
            "REGION",
            "MODEL_CLASS",
            "THREE_CLASS",
            "OPEN_TARGET",
            "P_SRKW",
        ]
    ].copy()
    for name, probability in predictions.items():
        if not np.isfinite(probability).all():
            raise ValueError(f"Open-set OOF predictions incomplete for {name}")
        export[name] = probability
    export["EVALUATION_WEIGHT"] = dataset.open_weights
    export["LABEL_PROPENSITY_WEIGHT"] = propensity_weight

    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    for name in OPEN_MODELS:
        predicted_other, thresholds = _crossfit_open_threshold(
            dataset.known, predictions[name], target_other_recall=0.90
        )
        threshold_export = thresholds.copy()
        threshold_export.insert(0, "model", name)
        threshold_rows.append(threshold_export)
        metric_rows.append(
            {
                "model": name,
                **binary_metrics(target, predictions[name], dataset.open_weights),
                "other_recall": float(recall_score(target == 0, predicted_other)),
                "modeled_retention": float(recall_score(target == 1, ~predicted_other)),
                "balanced_accuracy_at_recall_threshold": float(
                    balanced_accuracy_score(target, (~predicted_other).astype(int))
                ),
                "known_other_n": int((target == 0).sum()),
            }
        )
    return export, pd.DataFrame(metric_rows), pd.concat(threshold_rows, ignore_index=True)


def _binary_propensity_sensitivity(
    dataset: OpenSetDataset,
    paths: ReleasePaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = load_strategy_dataset(paths)
    propensity_lookup = pd.Series(
        dataset.labeled_propensity_oof,
        index=dataset.reference["ENCOUNTER_ID"].astype(str),
    )
    propensity = propensity_lookup.reindex(
        strategy.frame["ENCOUNTER_ID"].astype(str)
    ).to_numpy(float)
    ipw = propensity_weights(propensity)
    target = strategy.frame["Y_TRUE"].to_numpy(int)
    groups = strategy.frame["ENCOUNTER_ID"].astype(str).to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 980)
    predictions = np.full(len(target), np.nan)
    for fold, (train, test) in enumerate(folds.split(strategy.features, target, groups), start=1):
        model = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
                ),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.05,
                        max_iter=3000,
                        random_state=RANDOM_SEED + 980 + fold,
                    ),
                ),
            ]
        )
        model.fit(
            strategy.features.iloc[train],
            target[train],
            model__sample_weight=strategy.weights[train] * ipw[train],
        )
        predictions[test] = model.predict_proba(strategy.features.iloc[test])[:, 1]
    export = strategy.frame[
        ["OBSERVATION_ID", "ENCOUNTER_ID", "SOURCE", "ERA", "REGION", "Y_TRUE"]
    ].copy()
    export["current_model"] = strategy.features["current_model"]
    export["safe_transport_baseline"] = strategy.features[
        "constrained_current_transport_blend"
    ]
    export["propensity_corrected_logistic"] = predictions
    export["LABEL_PROPENSITY_WEIGHT"] = ipw
    metrics = pd.DataFrame(
        [
            {
                "evaluation_weight": "observed_natural_prevalence",
                "model": name,
                **binary_metrics(target, export[name], strategy.weights),
            }
            for name in (
                "current_model",
                "safe_transport_baseline",
                "propensity_corrected_logistic",
            )
        ]
        + [
            {
                "evaluation_weight": "inverse_label_propensity_sensitivity",
                "model": name,
                **binary_metrics(target, export[name], strategy.weights * ipw),
            }
            for name in (
                "current_model",
                "safe_transport_baseline",
                "propensity_corrected_logistic",
            )
        ]
    )
    return metrics, export


def _select_active_sample(
    dataset: OpenSetDataset,
    paths: ReleasePaths,
    best_model_name: str,
    *,
    sample_size: int = 200,
) -> pd.DataFrame:
    target = dataset.known["OPEN_TARGET"].to_numpy(int)
    propensity_weight = _known_propensity_weights(dataset)
    models = _fit_open_models(
        dataset.known_features,
        dataset.known,
        target,
        dataset.open_weights,
        propensity_weight,
        seed=RANDOM_SEED + 1100,
    )
    unknown = dataset.reference.loc[dataset.reference["TARGET_FAMILY"].eq("EXCLUDED")].copy()
    unknown_positions = unknown.index.to_numpy(int)
    unknown_features = dataset.reference_features.iloc[unknown_positions]
    open_probability = _predict_open_models(
        models, unknown_features, unknown
    )[best_model_name]

    propensity_model = _new_propensity_model(seed=RANDOM_SEED + 1101)
    propensity_target = dataset.reference["TARGET_FAMILY"].isin(["MODELED", "OTHER"]).astype(int)
    propensity_model.fit(_propensity_features(dataset.reference), propensity_target)
    label_probability = propensity_model.predict_proba(_propensity_features(unknown))[:, 1]

    diagnostics = pd.read_parquet(
        paths.diagnostics,
        columns=["OBSERVATION_ID", "P_SRKW", "OOD_MARGIN", "EVIDENCE_REGIME"],
    )
    unknown = unknown.merge(diagnostics, on="OBSERVATION_ID", how="left", validate="one_to_one")
    conditional = pd.to_numeric(unknown["P_SRKW"], errors="coerce").fillna(0.5).to_numpy(float)
    open_entropy = -(
        open_probability * np.log(np.clip(open_probability, 1e-9, 1.0))
        + (1 - open_probability) * np.log(np.clip(1 - open_probability, 1e-9, 1.0))
    ) / np.log(2.0)
    binary_uncertainty = 1.0 - np.abs(2.0 * conditional - 1.0)
    source_counts = unknown["SOURCE"].value_counts()
    rarity = 1.0 / np.sqrt(unknown["SOURCE"].map(source_counts).to_numpy(float))
    rarity /= rarity.max()
    unknown["P_MODELED_DIAGNOSTIC"] = open_probability
    unknown["P_OTHER_DIAGNOSTIC"] = 1.0 - open_probability
    unknown["P_LABEL_AVAILABLE_DIAGNOSTIC"] = label_probability
    unknown["ACTIVE_REVIEW_PRIORITY"] = (
        0.40 * open_entropy
        + 0.20 * binary_uncertainty
        + 0.25 * (1.0 - label_probability)
        + 0.15 * rarity
    )
    unknown["STRATUM_RANK"] = unknown.groupby(["SOURCE", "ERA"])[
        "ACTIVE_REVIEW_PRIORITY"
    ].rank(method="first", ascending=False)
    selected = unknown.sort_values(
        ["STRATUM_RANK", "ACTIVE_REVIEW_PRIORITY", "OBSERVATION_ID"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(min(sample_size, len(unknown)))
    selected = selected.sort_values(
        ["ACTIVE_REVIEW_PRIORITY", "OBSERVATION_ID"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected["REVIEW_STATUS"] = "UNREVIEWED"
    selected["SOFT_COUNT_ELIGIBLE"] = False
    selected["EXPECTED_UNKNOWN_COUNT"] = 1.0
    return selected[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "SOURCE",
            "ERA",
            "REGION",
            "LATITUDE",
            "LONGITUDE",
            "OBSERVATION_QUALITY_TIER",
            "SOURCE_TIME_PRECISION",
            "COORDINATE_UNCERTAINTY_M",
            "P_MODELED_DIAGNOSTIC",
            "P_OTHER_DIAGNOSTIC",
            "P_SRKW",
            "P_LABEL_AVAILABLE_DIAGNOSTIC",
            "OOD_MARGIN",
            "EVIDENCE_REGIME",
            "ACTIVE_REVIEW_PRIORITY",
            "REVIEW_STATUS",
            "SOFT_COUNT_ELIGIBLE",
            "EXPECTED_UNKNOWN_COUNT",
        ]
    ].reset_index(drop=True)


def _hierarchical_multiclass(
    open_predictions: pd.DataFrame,
    best_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeled_probability = np.clip(open_predictions[best_model].to_numpy(float), 0.0, 1.0)
    conditional_srkw = np.clip(open_predictions["P_SRKW"].to_numpy(float), 0.0, 1.0)
    probabilities = np.column_stack(
        [
            modeled_probability * conditional_srkw,
            modeled_probability * (1.0 - conditional_srkw),
            1.0 - modeled_probability,
        ]
    )
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Hierarchical open-set probabilities do not conserve unit mass")
    forced_binary = np.column_stack(
        [conditional_srkw, 1.0 - conditional_srkw, np.zeros(len(conditional_srkw))]
    )
    weights = open_predictions["EVALUATION_WEIGHT"].to_numpy(float)
    comparison = pd.DataFrame(
        [
            {
                "model": "current_forced_binary",
                **_multiclass_metrics(open_predictions["THREE_CLASS"], forced_binary, weights),
            },
            {
                "model": f"hierarchical_{best_model}",
                **_multiclass_metrics(open_predictions["THREE_CLASS"], probabilities, weights),
            },
        ]
    )
    export = open_predictions[
        ["OBSERVATION_ID", "ENCOUNTER_ID", "THREE_CLASS", "OPEN_TARGET"]
    ].copy()
    export["P_SRKW_UNCONDITIONAL"] = probabilities[:, 0]
    export["P_TRANSIENT_UNCONDITIONAL"] = probabilities[:, 1]
    export["P_OTHER_UNCONDITIONAL"] = probabilities[:, 2]
    export["PROBABILITY_MASS"] = probabilities.sum(axis=1)
    export["SOFT_COUNT_CERTIFIED"] = False
    export["EXPECTED_UNKNOWN_COUNT_IF_UNLABELED"] = 1.0
    return comparison, export


def run_open_set_missingness_experiment(
    output_dir: Path,
    paths: ReleasePaths | None = None,
) -> dict[str, Any]:
    paths = paths or resolve_release_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_open_set_dataset(paths)
    open_predictions, open_metrics, open_thresholds = crossfit_open_models(dataset)
    calibrated = open_metrics.loc[
        open_metrics["other_recall"].ge(0.90)
        & open_metrics["calibration_intercept"].abs().le(0.10)
        & open_metrics["calibration_slope"].between(0.8, 1.2)
        & open_metrics["equal_mass_ece_10"].le(0.03)
    ]
    selection_pool = calibrated if not calibrated.empty else open_metrics
    best_model = str(selection_pool.sort_values(["log_loss", "brier"]).iloc[0]["model"])
    multiclass, unconditional = _hierarchical_multiclass(open_predictions, best_model)
    binary_sensitivity, binary_predictions = _binary_propensity_sensitivity(dataset, paths)
    active_sample = _select_active_sample(dataset, paths, best_model)

    outputs = {
        "open_set_model_comparison.csv": open_metrics,
        "open_set_oof_predictions.csv": open_predictions,
        "open_set_thresholds.csv": open_thresholds,
        "hierarchical_multiclass_comparison.csv": multiclass,
        "hierarchical_unconditional_oof.csv": unconditional,
        "label_propensity_by_source.csv": dataset.propensity_by_source,
        "binary_propensity_sensitivity.csv": binary_sensitivity,
        "binary_propensity_oof_predictions.csv": binary_predictions,
        "active_label_audit_sample.csv": active_sample,
        "open_set_seascape_coverage.csv": dataset.seascape_coverage,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)
    write_json(output_dir / "label_propensity_metrics.json", dataset.propensity_metrics)

    known_other_n = int((dataset.known["OPEN_TARGET"] == 0).sum())
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "experiment": "hierarchical open-set and source-dependent label-missingness",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_id": paths.release_id,
            "snapshot_id": paths.snapshot_id,
            "release_manifest": paths.manifest_path,
            "observation_sha256": checksum_path(paths.observations),
            "diagnostics_sha256": checksum_path(paths.diagnostics),
            "known_other_encounter_n": known_other_n,
            "best_diagnostic_open_model": best_model,
            "p_other_semantics": (
                "biological non-SRKW/non-Transient probability; never an abstention flag"
            ),
            "abstention_semantics": (
                "unsupported unlabeled rows retain EXPECTED_UNKNOWN_COUNT=1 and contribute no "
                "diagnostic class probability to counts"
            ),
            "source_usage": (
                "label-propensity estimation, robustness groups, diagnostics, and audit-sample "
                "stratification only; never a biological open-set feature"
            ),
            "label_propensity_metrics": dataset.propensity_metrics,
            "seascape_artifacts": dataset.seascape_lineage,
            "active_label_sample_size": len(active_sample),
            "active_label_sample_status": "UNREVIEWED_INTERNAL_RESEARCH",
            "promotion_eligible": False,
            "promotion_blockers": [
                f"only {known_other_n} known-Other independent encounters are available",
                "no frozen blinded double-reviewed audit sample of unknown ecotypes exists",
                "missing-at-random assumption for inverse-propensity weighting is untestable",
                "open-set source and region holdouts have too few Other units for certification",
                "active-label priorities are acquisition suggestions, not generated labels",
            ],
            "outputs": [*outputs, "label_propensity_metrics.json"],
        },
    )
    return {
        "open_metrics": open_metrics,
        "open_predictions": open_predictions,
        "open_thresholds": open_thresholds,
        "multiclass": multiclass,
        "unconditional": unconditional,
        "propensity_metrics": dataset.propensity_metrics,
        "propensity_by_source": dataset.propensity_by_source,
        "binary_sensitivity": binary_sensitivity,
        "active_sample": active_sample,
        "best_model": best_model,
        "manifest_path": manifest_path,
    }


__all__ = [
    "OPEN_MODELS",
    "OpenSetDataset",
    "crossfit_label_propensity",
    "crossfit_open_models",
    "load_open_set_dataset",
    "propensity_weights",
    "run_open_set_missingness_experiment",
]
