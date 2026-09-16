"""Seasonal and static-seascape challengers for ecotype imputation.

This module is notebook-local and research-only.  It extends the learned
space-time transport experiment with deterministic calendar features and a
curated set of physically stable seascape features.  Every predictive model is
fit inside the reconstructed encounter-held-out outer folds; source is used
only for post-stratification, diagnostics, and conservative blend selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h3
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marine_mammal_toolkit.tools._core.checksums import checksum_path
from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputer import KillerWhaleImputer as SelectiveDateContextImputer

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
    _select_constrained_blend,
    _transport_design,
    _water_graph_sha256,
    build_transport_support,
    transport_kernel_grid,
)


@dataclass(frozen=True)
class SeascapeSpec:
    name: str
    relative_path: str
    resolution: int
    columns: tuple[str, ...]


STATIC_SEASCAPE_SPECS = (
    SeascapeSpec(
        "bathymetry",
        "seafloor_physiography/bathymetry/BATHYMETRY_RES_6.parquet",
        6,
        (
            "BATHYMETRY",
            "BATHYMETRY_MEDIAN",
            "BATHYMETRY_STD",
            "BATHYMETRY_RANGE",
            "BATHYMETRY_LOCAL_ANOMALY",
            "BATHYMETRY_Q10",
            "BATHYMETRY_Q90",
            "DISTANCE_TO_ISOBATH_50_M",
            "DISTANCE_TO_ISOBATH_100_M",
            "DISTANCE_TO_ISOBATH_200_M",
            "BATHYMETRY_FRAC_0_10_M",
            "BATHYMETRY_FRAC_10_30_M",
            "BATHYMETRY_FRAC_30_50_M",
            "BATHYMETRY_FRAC_50_100_M",
            "BATHYMETRY_FRAC_100_200_M",
            "BATHYMETRY_FRAC_OVER_200_M",
        ),
    ),
    SeascapeSpec(
        "geomorphometry",
        "seafloor_physiography/geomorphometry/GEOMORPHOMETRY_RES_8.parquet",
        8,
        (
            "SLOPE",
            "TERRAIN_POSITION",
            "CURVATURE",
            "RELIEF",
            "RUGGEDNESS",
            "EASTNESS",
            "NORTHNESS",
            "SLOPE_MEAN_RING_1",
            "SLOPE_MEAN_RING_2",
            "SLOPE_MEAN_RING_4",
            "TERRAIN_POSITION_RING_1_Z",
            "TERRAIN_POSITION_RING_2_Z",
            "TERRAIN_POSITION_RING_4_Z",
            "LOCAL_RELIEF_RING_1_M",
            "LOCAL_RELIEF_RING_2_M",
            "LOCAL_RELIEF_RING_4_M",
            "VECTOR_RUGGEDNESS_RING_1",
            "VECTOR_RUGGEDNESS_RING_2",
            "VECTOR_RUGGEDNESS_RING_4",
            "POSITIVE_OPENNESS_DEG",
            "NEGATIVE_OPENNESS_DEG",
            "RIDGE_INDEX",
            "VALLEY_INDEX",
        ),
    ),
    SeascapeSpec(
        "geomorphic_units",
        "seafloor_physiography/geomorphic_units/GEOMORPHIC_UNITS_RES_8.parquet",
        8,
        (
            "BROAD_TERRAIN_POSITION_M",
            "BROAD_TERRAIN_POSITION_Z",
            "DIRECTIONAL_ANISOTROPY",
            "CANYON_DENSITY",
            "DISTANCE_TO_SHELF_M",
            "DISTANCE_TO_SHELF_BREAK_M",
            "DISTANCE_TO_SLOPE_M",
            "DISTANCE_TO_BANK_OR_SHOAL_M",
            "DISTANCE_TO_BASIN_OR_DEPRESSION_M",
            "DISTANCE_TO_CANYON_AXIS_M",
            "DISTANCE_TO_CHANNEL_M",
            "DISTANCE_TO_SILL_M",
            "PROPORTION_SHELF",
            "PROPORTION_SHELF_BREAK",
            "PROPORTION_SLOPE",
            "PROPORTION_BANK_OR_SHOAL",
            "PROPORTION_BASIN_OR_DEPRESSION",
            "PROPORTION_CANYON_AXIS",
            "PROPORTION_CHANNEL",
            "PROPORTION_SILL",
        ),
    ),
    SeascapeSpec(
        "shoreline_proximity",
        "coastal_configuration/shoreline_proximity/SHORELINE_PROXIMITY_RES_8.parquet",
        8,
        ("SHORELINE_DISTANCE_M", "WATER_NETWORK_DISTANCE_M", "OPEN_OCEAN_INDEX"),
    ),
    SeascapeSpec(
        "exposure_enclosure",
        "coastal_configuration/exposure_and_enclosure/EXPOSURE_AND_ENCLOSURE_RES_8.parquet",
        8,
        (
            "OPENNESS_TO_OCEAN_INDEX",
            "ENCLOSURE_INDEX",
            "EMBAYMENT_INDEX",
            "DISTANCE_TO_OPEN_WATER_M",
            "OPEN_WATER_ANGULAR_APERTURE_DEG",
        ),
    ),
    SeascapeSpec(
        "waterbody_morphometry",
        "coastal_configuration/waterbody_morphometry/WATERBODY_MORPHOMETRY_RES_8.parquet",
        8,
        (
            "LOCAL_WATERBODY_WIDTH_M",
            "WATERBODY_WIDTH_MEAN_M",
            "WATERBODY_WIDTH_STD_M",
            "DISTANCE_TO_OPPOSITE_SHORE_M",
            "CONSTRICTION_INDEX",
            "LOCAL_WATERSPACE_AREA_KM2",
            "LOCAL_WATERSPACE_COMPACTNESS",
            "LOCAL_WATERSPACE_BRANCH_COUNT",
            "DISTANCE_TO_CONSTRICTED_PASSAGE_M",
            "DISTANCE_TO_SILL_CANDIDATE_M",
        ),
    ),
    SeascapeSpec(
        "shoreline_character",
        "coastal_configuration/shoreline_characterization/SHORELINE_CHARACTERIZATION_RES_6.parquet",
        6,
        (
            "SHORELINE_CLASSIFIED_COVERAGE_FRAC",
            "ROCKY_SHORE_FRAC",
            "SANDY_SHORE_FRAC",
            "GRAVEL_SHORE_FRAC",
            "CLIFF_SHORE_FRAC",
            "BLUFF_SHORE_FRAC",
            "DELTAIC_SHORE_FRAC",
            "ESTUARINE_SHORE_FRAC",
            "WATER_NETWORK_DISTANCE_TO_ROCKY_SHORE_M",
            "WATER_NETWORK_DISTANCE_TO_SANDY_SHORE_M",
            "WATER_NETWORK_DISTANCE_TO_ESTUARINE_SHORE_M",
        ),
    ),
    SeascapeSpec(
        "estuarine_connectivity",
        "hydrologic_connectivity/estuarine_connectivity/ESTUARINE_CONNECTIVITY_RES_8.parquet",
        8,
        ("DISTANCE_TO_ESTUARY_M", "WATER_NETWORK_DISTANCE_TO_ESTUARY_M"),
    ),
    SeascapeSpec(
        "fluvial_connectivity",
        "hydrologic_connectivity/fluvial_connectivity/FLUVIAL_CONNECTIVITY_RES_8.parquet",
        8,
        (
            "WATER_NETWORK_DISTANCE_TO_FLUVIAL_MOUTH_M",
            "EUCLIDEAN_DISTANCE_TO_FLUVIAL_MOUTH_M",
            "FLUVIAL_PATH_DETOUR_M",
            "FLUVIAL_PATH_DETOUR_RATIO",
            "FLUVIAL_MOUTH_REACHABLE",
            "CONNECTED_STRAHLER_ORDER",
            "CONNECTED_UPSTREAM_NETWORK_LENGTH_KM",
            "CONNECTED_UPSTREAM_DISTANCE_KM",
            "CONNECTED_UPSTREAM_DRAINAGE_AREA_KM2",
        ),
    ),
    SeascapeSpec(
        "benthic_substrate",
        "benthic_substrate/classification/BENTHIC_SUBSTRATE_CLASSIFICATION_RES_6.parquet",
        6,
        (
            "SUBSTRATE_ROCK_FRAC",
            "SUBSTRATE_BOULDER_FRAC",
            "SUBSTRATE_COBBLE_FRAC",
            "SUBSTRATE_GRAVEL_FRAC",
            "SUBSTRATE_SAND_FRAC",
            "SUBSTRATE_MUD_FRAC",
            "SUBSTRATE_MIXED_FRAC",
            "SUBSTRATE_INTERPOLATED_COVERAGE_FRAC",
            "SUBSTRATE_HARD_SUBSTRATE_FRAC",
            "SUBSTRATE_HETEROGENEITY",
            "SUBSTRATE_DISTANCE_TO_HARD_SUBSTRATE_M",
        ),
    ),
    SeascapeSpec(
        "bottom_hardness",
        "benthic_substrate/bottom_hardness/BOTTOM_HARDNESS_RES_6.parquet",
        6,
        (
            "BOTTOM_HARDNESS_INDEX",
            "CONSOLIDATED_SUBSTRATE_FRAC",
            "EXPOSED_ROCK_FRAC",
            "MODELED_HARD_SUBSTRATE_FRAC",
            "DISTANCE_TO_MODELED_HARD_SUBSTRATE_M",
        ),
    ),
)


def build_seasonal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return smooth calendar features without using event-year or source."""

    dates = pd.to_datetime(frame["SIGHTING_DATE"], errors="raise")
    day_of_year = dates.dt.dayofyear.to_numpy(float)
    days_in_year = np.where(dates.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    phase = 2.0 * np.pi * (day_of_year - 1.0) / days_in_year
    result: dict[str, np.ndarray] = {}
    for harmonic in range(1, 5):
        result[f"season__sin_{harmonic}"] = np.sin(harmonic * phase)
        result[f"season__cos_{harmonic}"] = np.cos(harmonic * phase)

    latitude_radians = np.deg2rad(frame["LATITUDE"].to_numpy(float))

    def daylight_hours(day: np.ndarray) -> np.ndarray:
        declination = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (284.0 + day) / 365.2425)
        cosine_hour_angle = np.clip(-np.tan(latitude_radians) * np.tan(declination), -1.0, 1.0)
        return 24.0 * np.arccos(cosine_hour_angle) / np.pi

    daylight = daylight_hours(day_of_year)
    result["season__daylight_hours"] = daylight
    result["season__daylight_change_hours"] = daylight_hours(day_of_year + 1.0) - daylight
    circular_winter = np.minimum(
        np.abs(day_of_year - 355.0), days_in_year - np.abs(day_of_year - 355.0)
    )
    circular_summer = np.minimum(
        np.abs(day_of_year - 172.0), days_in_year - np.abs(day_of_year - 172.0)
    )
    result["season__days_from_winter_solstice"] = circular_winter
    result["season__days_from_summer_solstice"] = circular_summer

    month = dates.dt.month.to_numpy(int)
    result["season__winter"] = np.isin(month, (12, 1, 2)).astype(float)
    result["season__spring"] = np.isin(month, (3, 4, 5)).astype(float)
    result["season__summer"] = np.isin(month, (6, 7, 8)).astype(float)
    result["season__autumn"] = np.isin(month, (9, 10, 11)).astype(float)
    seasonal = pd.DataFrame(result, index=frame.index)
    if not np.isfinite(seasonal.to_numpy(float)).all():
        raise ValueError("Seasonal feature construction produced non-finite values")
    return seasonal


def load_static_seascape_features(
    frame: pd.DataFrame,
    seascape_root: Path,
    *,
    specs: Sequence[SeascapeSpec] = STATIC_SEASCAPE_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Join native R6/R8 physical seascape values to sighting coordinates."""

    latitude = frame["LATITUDE"].to_numpy(float)
    longitude = frame["LONGITUDE"].to_numpy(float)
    cells_by_resolution = {
        resolution: np.asarray(
            [
                h3.latlng_to_cell(lat, lon, resolution)
                for lat, lon in zip(latitude, longitude, strict=True)
            ]
        )
        for resolution in sorted({spec.resolution for spec in specs})
    }
    feature_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for spec in specs:
        path = seascape_root / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required static seascape artifact is missing: {path}")
        table = pd.read_parquet(path, columns=["H3_INDEX", *spec.columns])
        if table["H3_INDEX"].duplicated().any():
            raise ValueError(f"{spec.name} contains duplicate H3_INDEX rows")
        missing_columns = sorted(set(spec.columns) - set(table.columns))
        if missing_columns:
            raise ValueError(f"{spec.name} is missing configured columns: {missing_columns}")
        table["H3_INDEX"] = table["H3_INDEX"].astype(str)
        indexed = table.set_index("H3_INDEX")
        target_cells = cells_by_resolution[spec.resolution]
        matched = pd.Index(target_cells).isin(indexed.index)
        joined = indexed.reindex(target_cells).reset_index(drop=True)
        joined.index = frame.index
        numeric = joined.loc[:, spec.columns].apply(pd.to_numeric, errors="coerce")
        numeric.columns = [f"seascape__{spec.name}__{column.lower()}" for column in spec.columns]
        numeric[f"seascape__{spec.name}__cell_available"] = matched.astype(float)
        for column in tuple(numeric.columns):
            if numeric[column].isna().any():
                numeric[f"{column}__missing"] = numeric[column].isna().astype(float)
        feature_parts.append(numeric)
        finite_count = np.isfinite(numeric[list(numeric.columns[: len(spec.columns)])]).sum(axis=1)
        coverage_rows.append(
            {
                "product": spec.name,
                "resolution": spec.resolution,
                "artifact": spec.relative_path,
                "artifact_row_n": len(table),
                "feature_n": len(spec.columns),
                "target_n": len(frame),
                "matched_cell_n": int(matched.sum()),
                "matched_cell_rate": float(matched.mean()),
                "all_features_present_n": int((finite_count == len(spec.columns)).sum()),
                "all_features_present_rate": float((finite_count == len(spec.columns)).mean()),
            }
        )
        lineage.append(
            {
                "product": spec.name,
                "path": spec.relative_path,
                "resolution": spec.resolution,
                "sha256": checksum_path(path),
                "columns": list(spec.columns),
            }
        )
    features = pd.concat(feature_parts, axis=1)
    if features.columns.duplicated().any():
        raise ValueError("Static seascape feature names are not unique")
    return features, pd.DataFrame(coverage_rows), lineage


def _new_covariate_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    l1_ratio=1.0,
                    solver="liblinear",
                    max_iter=3000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _select_covariate_model(
    design: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 500)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for c_value in (0.001, 0.005, 0.02, 0.1):
        probability = np.full(len(target), np.nan, dtype=float)
        losses: list[float] = []
        for fit_indices, score_indices in splitter.split(design, target, groups):
            model = _new_covariate_model(c_value)
            model.fit(
                design.iloc[fit_indices],
                target[fit_indices],
                model__sample_weight=weights[fit_indices],
            )
            fold_probability = model.predict_proba(design.iloc[score_indices])[:, 1]
            probability[score_indices] = fold_probability
            losses.append(
                float(
                    log_loss(
                        target[score_indices],
                        fold_probability,
                        sample_weight=weights[score_indices],
                        labels=[0, 1],
                    )
                )
            )
        if not np.isfinite(probability).all():
            raise ValueError("Covariate tuning left rows without predictions")
        candidates.append((float(np.mean(losses)), c_value, probability))
    score, c_value, probability = min(candidates, key=lambda item: (item[0], item[1]))
    return c_value, score, probability


def _select_multistratum_constrained_blend(
    current_probability: np.ndarray,
    challenger_probability: np.ndarray,
    target: np.ndarray,
    strata: pd.DataFrame,
    weights: np.ndarray,
    *,
    minimum_stratum_n: int = 50,
    maximum_brier_degradation: float = 0.10,
) -> tuple[float, float, float]:
    """Select a blend using only outer-training source/era/region guardrails."""

    required_columns = {"SOURCE", "ERA", "REGION"}
    if not required_columns.issubset(strata.columns):
        raise ValueError(
            f"Blend strata are missing {sorted(required_columns - set(strata.columns))}"
        )
    candidates: list[tuple[float, float, float]] = []
    for current_weight in (0.0, 0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 1.0):
        probability = (
            current_weight * current_probability + (1.0 - current_weight) * challenger_probability
        )
        degradations: list[float] = []
        for column in ("SOURCE", "ERA", "REGION"):
            values = strata[column].astype(str).to_numpy()
            for value in np.unique(values):
                mask = values == value
                if int(mask.sum()) < minimum_stratum_n:
                    continue
                baseline = float(
                    brier_score_loss(
                        target[mask], current_probability[mask], sample_weight=weights[mask]
                    )
                )
                challenger = float(
                    brier_score_loss(target[mask], probability[mask], sample_weight=weights[mask])
                )
                degradations.append(challenger / baseline - 1.0 if baseline > 0 else np.inf)
        maximum_degradation = max(degradations, default=0.0)
        if maximum_degradation <= maximum_brier_degradation + 1e-12:
            score = float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))
            candidates.append((score, current_weight, maximum_degradation))
    if not candidates:
        raise ValueError("The incumbent-only blend should satisfy every training guardrail")
    score, current_weight, maximum_degradation = min(
        candidates, key=lambda item: (item[0], -item[1])
    )
    return current_weight, score, maximum_degradation


def _base_score_design(
    current_probability: np.ndarray,
    transport_probability: np.ndarray,
    transport_support: pd.DataFrame,
) -> pd.DataFrame:
    total_support_columns = [
        column
        for column in transport_support.columns
        if "__srkw__" in column or "__transient__" in column
    ]
    has_support = transport_support[total_support_columns].sum(axis=1).gt(0).astype(float)
    return pd.DataFrame(
        {
            "score__current_logit": logit(np.clip(current_probability, 1e-6, 1 - 1e-6)),
            "score__transport_logit": logit(np.clip(transport_probability, 1e-6, 1 - 1e-6)),
            "score__transport_has_support": has_support.to_numpy(float),
        },
        index=transport_support.index,
    )


def build_covariate_design(
    current_probability: np.ndarray,
    transport_probability: np.ndarray,
    transport_support: pd.DataFrame,
    seasonal: pd.DataFrame,
    seascape: pd.DataFrame,
    *,
    include_transport: bool,
    include_season: bool,
    include_seascape: bool,
    include_season_interactions: bool,
) -> pd.DataFrame:
    base = _base_score_design(current_probability, transport_probability, transport_support)
    if not include_transport:
        base = base.drop(columns=["score__transport_logit", "score__transport_has_support"])
    parts = [base]
    if include_season:
        parts.append(seasonal)
    if include_seascape:
        parts.append(seascape)
    design = pd.concat(parts, axis=1)
    if include_transport and include_season_interactions:
        transport_logit = base["score__transport_logit"]
        for column in [item for item in seasonal.columns if "__sin_" in item or "__cos_" in item]:
            design[f"interaction__transport__{column}"] = transport_logit * seasonal[column]
    if include_seascape and include_season_interactions:
        for seasonal_column in ("season__sin_1", "season__cos_1"):
            values = seasonal[seasonal_column]
            for seascape_column in [
                item
                for item in seascape.columns
                if not item.endswith("__missing") and not item.endswith("__cell_available")
            ]:
                design[f"interaction__{seasonal_column}__{seascape_column}"] = (
                    values * seascape[seascape_column]
                )
    if design.columns.duplicated().any():
        raise ValueError("Covariate design contains duplicate feature names")
    return design


def _coefficient_rows(
    model: Pipeline,
    columns: Sequence[str],
    *,
    fold: int,
    variant: str,
    limit: int = 50,
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


def _stratified_metrics(
    frame: pd.DataFrame,
    model_names: Sequence[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratification, column in (("SOURCE", "SOURCE"), ("ERA", "ERA"), ("REGION", "REGION")):
        for value, indices in frame.groupby(column, sort=True).groups.items():
            positions = frame.index.get_indexer(indices)
            for model_name in model_names:
                rows.append(
                    {
                        "stratification": stratification,
                        "stratum": str(value),
                        "model": model_name,
                        **binary_metrics(
                            frame.loc[indices, "Y_TRUE"],
                            frame.loc[indices, model_name],
                            weights[positions],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _gate_table(
    metrics: pd.DataFrame,
    model_names: Sequence[str],
    *,
    baseline_model: str = "current_model",
    minimum_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = metrics.loc[metrics["model"].eq(baseline_model)].set_index(
        ["stratification", "stratum"]
    )
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        challenger = metrics.loc[metrics["model"].eq(model_name)].set_index(
            ["stratification", "stratum"]
        )
        for key in current.index:
            current_row = current.loc[key]
            challenger_row = challenger.loc[key]
            current_brier = float(current_row["brier"])
            challenger_brier = float(challenger_row["brier"])
            degradation = challenger_brier / current_brier - 1 if current_brier > 0 else np.inf
            independent_n = int(current_row["n"])
            rows.append(
                {
                    "model": model_name,
                    "baseline_model": baseline_model,
                    "stratification": key[0],
                    "stratum": key[1],
                    "independent_n": independent_n,
                    "current_brier": current_brier,
                    "challenger_brier": challenger_brier,
                    "relative_brier_degradation": degradation,
                    "required_gate": independent_n >= minimum_n,
                    "passes_10pct_degradation_gate": independent_n < minimum_n
                    or degradation <= 0.10,
                }
            )
    gates = pd.DataFrame(rows)
    summary = (
        gates.loc[gates["required_gate"]]
        .groupby(["model", "stratification"], as_index=False)
        .agg(
            required_stratum_n=("stratum", "nunique"),
            failed_stratum_n=("passes_10pct_degradation_gate", lambda values: int((~values).sum())),
            maximum_relative_brier_degradation=("relative_brier_degradation", "max"),
        )
    )
    summary["all_required_strata_pass"] = summary["failed_stratum_n"].eq(0)
    return gates, summary


def run_covariate_experiment(
    output_dir: Path,
    paths: ReleasePaths | None = None,
) -> dict[str, Any]:
    paths = paths or resolve_release_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    imputer = SelectiveDateContextImputer.load(paths.model_dir / "ecotype_imputer.joblib")
    oof = pd.read_csv(paths.model_dir / "metrics/encounter/oof_predictions.csv")
    oof["MODEL_CLASS"] = oof["MODEL_CLASS"].str.upper()
    if oof["ENCOUNTER_ID"].duplicated().any():
        raise ValueError("Covariate evaluation requires independent encounter representatives")

    reference = _encounter_reference(paths)
    binary_reference = reference.loc[reference["ECOTYPE_DETAIL"].isin(["SRKW", "TRANSIENT"])].copy()
    binary_reference["MODEL_CLASS"] = binary_reference["ECOTYPE_DETAIL"]
    weights = _poststratification_weights(oof, binary_reference, label_column="MODEL_CLASS")
    target = oof["Y_TRUE"].to_numpy(int)
    groups = oof["ENCOUNTER_ID"].astype(str).to_numpy()
    anchors = imputer.anchors_.copy()
    kernels = transport_kernel_grid()
    seasonal = build_seasonal_features(oof)
    seascape_root = paths.repo_root / "data/processed/domain/environmental_layer/seascape"
    seascape, seascape_coverage, seascape_lineage = load_static_seascape_features(
        oof, seascape_root
    )

    dates = pd.to_datetime(oof["SIGHTING_DATE"])
    oof["ERA"] = pd.cut(
        dates.dt.year,
        bins=[1979, 1999, 2009, 2019, np.inf],
        labels=["1980-1999", "2000-2009", "2010-2019", "2020+"],
    ).astype(str)
    oof["REGION"] = [
        h3.latlng_to_cell(lat, lon, 4)
        for lat, lon in zip(oof["LATITUDE"], oof["LONGITUDE"], strict=True)
    ]

    variant_settings = {
        "seasonal_current_raw": dict(
            include_transport=False,
            include_season=True,
            include_seascape=False,
            include_season_interactions=False,
        ),
        "seasonal_transport_raw": dict(
            include_transport=True,
            include_season=True,
            include_seascape=False,
            include_season_interactions=True,
        ),
        "static_seascape_transport_raw": dict(
            include_transport=True,
            include_season=False,
            include_seascape=True,
            include_season_interactions=False,
        ),
        "seasonal_seascape_transport_raw": dict(
            include_transport=True,
            include_season=True,
            include_seascape=True,
            include_season_interactions=True,
        ),
    }
    prediction_names = [
        "constrained_current_transport_blend",
        *variant_settings,
        "gated_seasonal_transport_blend",
        "gated_static_seascape_transport_blend",
        "gated_seasonal_seascape_transport_blend",
    ]
    predictions = {name: np.full(len(oof), np.nan, dtype=float) for name in prediction_names}
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
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
        from transport_experiment import _new_sparse_logistic, _select_sparse_logistic

        selected_transport_c, transport_inner_loss, transport_train_oof = _select_sparse_logistic(
            train_transport_design,
            target[train_indices],
            groups[train_indices],
            weights[train_indices],
        )
        transport_model = _new_sparse_logistic(selected_transport_c)
        transport_model.fit(
            train_transport_design,
            target[train_indices],
            model__sample_weight=weights[train_indices],
        )
        transport_test_probability = transport_model.predict_proba(test_transport_design)[:, 1]
        current_train = oof.iloc[train_indices]["P_SRKW"].to_numpy(float)
        current_test = oof.iloc[test_indices]["P_SRKW"].to_numpy(float)
        transport_current_weight, transport_blend_loss, transport_max_degradation = (
            _select_constrained_blend(
                current_train,
                transport_train_oof,
                target[train_indices],
                oof.iloc[train_indices]["SOURCE"].astype(str).to_numpy(),
                weights[train_indices],
            )
        )
        predictions["constrained_current_transport_blend"][test_indices] = (
            transport_current_weight * current_test
            + (1.0 - transport_current_weight) * transport_test_probability
        )

        train_oof_by_variant: dict[str, np.ndarray] = {}
        fold_detail: dict[str, Any] = {
            "outer_fold": fold,
            "train_n": len(train_indices),
            "test_n": len(test_indices),
            "transport_selected_c": selected_transport_c,
            "transport_inner_log_loss": transport_inner_loss,
            "transport_blend_current_weight": transport_current_weight,
            "transport_blend_inner_log_loss": transport_blend_loss,
            "transport_blend_max_source_brier_degradation": transport_max_degradation,
        }
        for variant, settings in variant_settings.items():
            train_design = build_covariate_design(
                current_train,
                transport_train_oof,
                transport_support.iloc[train_indices],
                seasonal.iloc[train_indices],
                seascape.iloc[train_indices],
                **settings,
            )
            test_design = build_covariate_design(
                current_test,
                transport_test_probability,
                transport_support.iloc[test_indices],
                seasonal.iloc[test_indices],
                seascape.iloc[test_indices],
                **settings,
            )
            selected_c, inner_loss, train_probability = _select_covariate_model(
                train_design,
                target[train_indices],
                groups[train_indices],
                weights[train_indices],
            )
            model = _new_covariate_model(selected_c)
            model.fit(
                train_design,
                target[train_indices],
                model__sample_weight=weights[train_indices],
            )
            predictions[variant][test_indices] = model.predict_proba(test_design)[:, 1]
            train_oof_by_variant[variant] = train_probability
            fold_detail[f"{variant}__selected_c"] = selected_c
            fold_detail[f"{variant}__inner_log_loss"] = inner_loss
            fold_detail[f"{variant}__feature_n"] = train_design.shape[1]
            fold_detail[f"{variant}__nonzero_coefficient_n"] = int(
                np.count_nonzero(model.named_steps["model"].coef_[0])
            )
            coefficient_rows.extend(
                _coefficient_rows(model, train_design.columns, fold=fold, variant=variant)
            )

        for raw_variant, gated_variant in (
            ("seasonal_transport_raw", "gated_seasonal_transport_blend"),
            ("static_seascape_transport_raw", "gated_static_seascape_transport_blend"),
            ("seasonal_seascape_transport_raw", "gated_seasonal_seascape_transport_blend"),
        ):
            current_weight, inner_loss, maximum_degradation = (
                _select_multistratum_constrained_blend(
                    current_train,
                    train_oof_by_variant[raw_variant],
                    target[train_indices],
                    oof.iloc[train_indices][["SOURCE", "ERA", "REGION"]],
                    weights[train_indices],
                )
            )
            predictions[gated_variant][test_indices] = (
                current_weight * current_test
                + (1.0 - current_weight) * predictions[raw_variant][test_indices]
            )
            fold_detail[f"{gated_variant}__current_weight"] = current_weight
            fold_detail[f"{gated_variant}__inner_log_loss"] = inner_loss
            fold_detail[f"{gated_variant}__max_stratum_brier_degradation"] = maximum_degradation
        fold_rows.append(fold_detail)

    oof["current_model"] = oof["P_SRKW"].to_numpy(float)
    for name, probability in predictions.items():
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"{name} did not produce finite unit-interval probabilities")
        oof[name] = probability

    model_names = ["current_model", *prediction_names]
    comparison = pd.DataFrame(
        [{"model": name, **binary_metrics(target, oof[name], weights)} for name in model_names]
    )
    stratified_metrics = _stratified_metrics(oof, model_names, weights)
    gates, gate_summary = _gate_table(stratified_metrics, prediction_names)
    pass_by_model = gate_summary.groupby("model")["all_required_strata_pass"].all()
    eligible = set(pass_by_model[pass_by_model].index)
    feature_candidates = [
        name for name in prediction_names if name != "constrained_current_transport_blend"
    ]
    eligible_feature_comparison = comparison.loc[
        comparison["model"].isin(eligible.intersection(feature_candidates))
    ]
    recommended_feature = (
        str(eligible_feature_comparison.sort_values(["log_loss", "brier"]).iloc[0]["model"])
        if not eligible_feature_comparison.empty
        else None
    )
    eligible_comparison = comparison.loc[comparison["model"].isin(eligible)]
    recommended = (
        str(eligible_comparison.sort_values(["log_loss", "brier"]).iloc[0]["model"])
        if not eligible_comparison.empty
        else "current_model"
    )
    bootstrap = {
        name: _paired_bootstrap(
            target,
            oof["current_model"].to_numpy(float),
            oof[name].to_numpy(float),
            weights,
        )
        for name in prediction_names
    }

    prediction_export = oof[
        [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "SOURCE",
            "MODEL_CLASS",
            "Y_TRUE",
            "ERA",
            "REGION",
            *model_names,
        ]
    ].copy()
    frames = {
        "covariate_comparison": comparison,
        "covariate_metrics_by_stratum": stratified_metrics,
        "covariate_stratum_gates": gates,
        "covariate_gate_summary": gate_summary,
        "covariate_fold_selections": pd.DataFrame(fold_rows),
        "covariate_top_coefficients": pd.DataFrame(coefficient_rows),
        "covariate_seascape_coverage": seascape_coverage,
        "covariate_transport_support_diagnostics": pd.DataFrame(support_rows),
        "covariate_oof_predictions": prediction_export,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    seascape_release_manifest = seascape_root / "seascape_release_manifest.json"
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "experiment": "seasonal and static physical seascape ecotype challengers",
            "release_id": paths.release_id,
            "snapshot_id": paths.snapshot_id,
            "release_manifest": paths.manifest_path,
            "model_sha256": imputer.training_summary_.get("MODEL_IDENTITY_SHA256"),
            "water_graph_sha256": _water_graph_sha256(imputer.feature_builder.marine_lookup.graph),
            "seascape_release_manifest": str(
                seascape_release_manifest.relative_to(paths.repo_root)
            ),
            "seascape_release_manifest_sha256": checksum_path(seascape_release_manifest),
            "seascape_artifacts": seascape_lineage,
            "seasonal_features": list(seasonal.columns),
            "static_feature_n": int(seascape.shape[1]),
            "outer_strategy": "reconstructed current encounter folds",
            "outer_folds": int(imputer.config.model.n_splits),
            "inner_folds": 3,
            "source_used_as_feature": False,
            "source_usage": "post-stratification, diagnostics, and blend guardrails only",
            "natural_prevalence_poststratification": "SOURCE x observed class encounter prevalence",
            "guardrails": {
                "minimum_independent_stratum_n": 100,
                "maximum_relative_brier_degradation": 0.10,
                "stratifications": ["SOURCE", "ERA", "H3_R4_REGION"],
            },
            "bootstrap_vs_current": bootstrap,
            "recommended_research_challenger": recommended,
            "recommended_feature_augmented_challenger": recommended_feature,
            "recommended_challenger_rule": (
                "lowest natural-prevalence log loss among all challengers passing every required "
                "source, era, and H3 R4 region Brier degradation gate"
            ),
            "excluded_covariates": {
                "kelp_seagrass_and_dynamic_habitat": "not time-frozen for historical sightings",
                "anthropogenic_features": "not time-frozen and may encode observer access",
                "source": "diagnostic and calibration stratum, never a biological predictor",
            },
            "promotion_eligible": False,
            "promotion_blockers": [
                "binary SRKW-versus-Transient target remains closed-set",
                "evaluation uses the current capped encounter sample",
                "only three configured random encounter outer folds are available",
                "rolling-origin and repeated spatial outer holdouts remain required",
                "static physical artifacts are not historical state snapshots",
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
        "recommended_feature_augmented_challenger": recommended_feature,
        "manifest_path": manifest_path,
    }


__all__ = [
    "STATIC_SEASCAPE_SPECS",
    "SeascapeSpec",
    "build_covariate_design",
    "build_seasonal_features",
    "load_static_seascape_features",
    "run_covariate_experiment",
]
