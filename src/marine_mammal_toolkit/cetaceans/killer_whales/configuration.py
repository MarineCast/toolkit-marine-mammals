from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import Field, PositiveInt, field_validator, model_validator

from marine_mammal_toolkit.tools._core.config import ConfigDocument
from marine_mammal_toolkit.tools._core.config.models import StrictConfig


from marine_mammal_toolkit.tools.schemas.sources import (
    BBox,
    GbifDatasetSettings,
    CwrAtlistMapSettings,
    SourceSettings,
)


class CollectionSettings(StrictConfig):
    overlap_days: int = Field(default=2, ge=0)
    sources: dict[
        Literal["twm", "acartia", "maplify", "inaturalist", "cwr", "gbif"],
        SourceSettings,
    ]

    @field_validator("sources")
    @classmethod
    def complete_source_selection(cls, value):
        """Omitted providers are disabled, never silently enabled."""
        return {
            name: value.get(name, SourceSettings(enabled=False))
            for name in ("twm", "acartia", "maplify", "inaturalist", "cwr", "gbif")
        }


class DeduplicationSettings(StrictConfig):
    timestamp_tolerance_minutes: int = Field(default=15, ge=0)
    distance_tolerance_m: float = Field(default=500.0, gt=0)


class KernelSettings(StrictConfig):
    radius_km: float = Field(default=100.0, gt=0)
    center_weight: float = Field(default=1.0, ge=0, le=1)
    outer_weight: float = Field(default=0.015, ge=0, le=1)
    exponent: float = Field(default=1.5, gt=0)
    combine: Literal["noisy_or"] = "noisy_or"

    @model_validator(mode="after")
    def validate_weights(self) -> "KernelSettings":
        if self.center_weight < self.outer_weight:
            raise ValueError(
                "center_weight must be greater than or equal to outer_weight"
            )
        return self


class ModelUniverseSettings(StrictConfig):
    polygon: Path
    membership_rule: Literal["representative_point"] = "representative_point"


class ImputationInputSettings(StrictConfig):
    observations: Path = Path(
        "data/processed/sightings/normalized/observations.parquet"
    )
    associations: Path | None = Path(
        "data/processed/sightings/normalized/associations.parquet"
    )
    water_network_config: Path = Path("config/data/environment_seascape.yaml")


class ImputationArtifactSettings(StrictConfig):
    models_dir: Path = Path("models/sighting_imputation")
    output: Path = Path("data/processed/sightings/imputed/imputed-sightings.parquet")


class ImputationFeatureSettings(StrictConfig):
    max_radius_km: float = Field(default=60.0, gt=0)
    distance_scale_km: float = Field(default=10.0, gt=0)
    time_scale_days: float = Field(default=4.0, gt=0)
    max_day_lag: int = Field(default=30, ge=0)
    support_unit_km: float = Field(default=2.5, gt=0)
    query_chunk_size: int = Field(default=2_000, gt=0)
    marine_fallback_to_haversine: bool = False
    max_target_snap_km: float = Field(default=5.0, ge=0)
    strong_local_support_floor: float = Field(default=0.01, ge=0)
    movement_speed_scale_km_per_day: float = Field(default=50.0, gt=0)
    maximum_feasible_km_per_day: float = Field(default=150.0, gt=0)
    marine_h3_resolution: Literal[6, 8] = 6
    temporal_windows: tuple[tuple[str, int, int], ...] = (
        ("same_day", 0, 0),
        ("past_1d", -1, -1),
        ("past_2_3d", -3, -2),
        ("past_4_7d", -7, -4),
        ("past_8_14d", -14, -8),
        ("past_15_30d", -30, -15),
        ("future_1d", 1, 1),
        ("future_2_3d", 2, 3),
        ("future_4_7d", 4, 7),
        ("future_8_14d", 8, 14),
        ("future_15_30d", 15, 30),
    )

    @model_validator(mode="after")
    def validate_temporal_windows(self) -> "ImputationFeatureSettings":
        names = [name for name, _, _ in self.temporal_windows]
        if not self.temporal_windows or len(names) != len(set(names)):
            raise ValueError(
                "Imputation temporal-window names must be nonempty and unique"
            )
        for name, lower, upper in self.temporal_windows:
            if lower > upper:
                raise ValueError(
                    f"Invalid imputation temporal window {name}: {lower} > {upper}"
                )
            if abs(lower) > self.max_day_lag or abs(upper) > self.max_day_lag:
                raise ValueError(
                    f"Imputation temporal window {name} exceeds max_day_lag={self.max_day_lag}"
                )
        return self


class ImputationModelSettings(StrictConfig):
    random_state: int = 42
    n_splits: int = Field(default=3, ge=2)
    model_kind: Literal["extra_trees", "hist_gradient_boosting", "logistic"] = (
        "extra_trees"
    )
    n_estimators: int = Field(default=400, gt=0)
    max_features: float = Field(default=0.70, gt=0, le=1)
    learning_rate: float = Field(default=0.05, gt=0)
    max_iter: int = Field(default=350, gt=0)
    max_leaf_nodes: int = Field(default=23, gt=1)
    min_samples_leaf: int = Field(default=20, gt=0)
    l2_regularization: float = Field(default=1.0, ge=0)
    logistic_c: float = Field(default=0.5, gt=0)
    calibration_method: Literal["auto", "sigmoid", "isotonic"] = "auto"
    conformal_alpha: float = Field(default=0.10, gt=0, lt=1)
    target_selective_error: float = Field(default=0.05, gt=0, lt=0.5)
    risk_confidence: float = Field(default=0.95, gt=0.5, lt=1)
    min_policy_examples: int = Field(default=20, gt=0)
    min_policy_accepts: int = Field(default=12, gt=0)
    nested_decision_splits: Literal[2] = 2
    ood_contamination: float = Field(default=0.01, ge=0, lt=0.5)
    other_support_veto_ratio: float = Field(default=1.05, gt=0)
    other_support_veto_floor: float = Field(default=0.20, ge=0)
    blocked_days: int = Field(default=7, gt=0)
    blocked_km: float = Field(default=20.0, gt=0)
    minimum_labeled_per_class: int = Field(default=30, gt=1)
    max_evaluation_per_class: int | None = Field(default=None, gt=1)
    ood_n_estimators: int = Field(default=100, gt=0)
    ood_min_regime_examples: int = Field(default=50, gt=0)
    max_training_majority_ratio: float = Field(default=10.0, ge=1)
    encounter_radius_km: float = Field(default=3.0, gt=0)
    encounter_association_radius_km: float = Field(default=20.0, gt=0)
    encounter_timestamp_tolerance_hours: float = Field(default=6.0, gt=0)
    use_regime_experts: bool = True
    min_expert_examples: int = Field(default=100, ge=20)
    enable_hyperparameter_tuning: bool = True
    tuning_max_candidates: int = Field(default=8, gt=0)
    tuning_n_jobs: int = Field(default=2, gt=0)
    enable_prediction_stability: bool = True
    maximum_stability_probability_shift: float = Field(default=0.20, gt=0, lt=0.5)
    hard_certification_min_accepts_per_class: int = Field(default=100, ge=100)
    enable_hard_label_release: bool = False
    hard_abstain_regimes: tuple[str, ...] = (
        "LOCATION_OFF_WATER",
        "MARINE_LOOKUP_MISSING",
        "WATER_BLOCKED",
        "MARINE_OUT_OF_RANGE",
        "NO_LOCAL_CANDIDATES",
        "OTHER_CONTEXT_ONLY",
    )
    final_calibration_strategy: Literal[
        "reconstruction", "encounter", "blocked", "purged_blocked"
    ] = "encounter"
    hard_label_certification_strategy: Literal["purged_blocked"] = "purged_blocked"


class ImputationSettings(StrictConfig):
    inputs: ImputationInputSettings = Field(default_factory=ImputationInputSettings)
    artifacts: ImputationArtifactSettings = Field(
        default_factory=ImputationArtifactSettings
    )
    regime: Literal["retrospective", "operational_eod"] = "retrospective"
    evaluate_strategies: tuple[
        Literal["reconstruction", "encounter", "blocked", "purged_blocked"], ...
    ] = ("reconstruction", "encounter", "blocked", "purged_blocked")
    feature: ImputationFeatureSettings = Field(
        default_factory=ImputationFeatureSettings
    )
    model: ImputationModelSettings = Field(default_factory=ImputationModelSettings)

    @field_validator("evaluate_strategies")
    @classmethod
    def validate_evaluation_strategies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("At least one imputation evaluation strategy is required")
        if len(value) != len(set(value)):
            raise ValueError("Imputation evaluation strategies must be unique")
        if "purged_blocked" not in value:
            raise ValueError(
                "purged_blocked evaluation is required for hard-label certification"
            )
        return value

    @model_validator(mode="after")
    def validate_certification_strategy(self) -> "ImputationSettings":
        if self.model.final_calibration_strategy not in self.evaluate_strategies:
            raise ValueError(
                "final_calibration_strategy must be included in evaluate_strategies"
            )
        if self.model.hard_label_certification_strategy not in self.evaluate_strategies:
            raise ValueError(
                "hard_label_certification_strategy must be included in evaluate_strategies"
            )
        return self


def _default_aggregation_chunks() -> dict[Literal["daily", "weekly"], PositiveInt]:
    return {"daily": 31, "weekly": 13}


class SightingsPipelineConfig(StrictConfig):
    @property
    def count_policy(self):
        from .observations.interpretation import COUNT_POLICY

        return COUNT_POLICY

    @property
    def observation_policy(self):
        from .observations.interpretation import OBSERVATION_POLICY

        return OBSERVATION_POLICY

    schema_version: Literal[6] = 6
    model_timezone: str = "America/Los_Angeles"
    full_area: BBox
    model_universes: dict[Literal["SRKW", "TRANSIENT"], ModelUniverseSettings]
    min_date: str = "1980-01-01"
    max_date: str | None = None
    water_network_config: Path = Path("config/data/environment_seascape.yaml")
    h3_resolutions: tuple[int, ...] = (4, 5, 6)
    frequencies: tuple[Literal["daily", "weekly"], ...] = ("daily", "weekly")
    aggregation_chunk_periods: dict[Literal["daily", "weekly"], PositiveInt] = Field(
        default_factory=_default_aggregation_chunks
    )
    collection: CollectionSettings
    deduplication: DeduplicationSettings = Field(default_factory=DeduplicationSettings)
    kernel: KernelSettings = Field(default_factory=KernelSettings)
    imputation: ImputationSettings = Field(default_factory=ImputationSettings)

    @field_validator("model_timezone")
    @classmethod
    def validate_model_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @field_validator("min_date")
    @classmethod
    def validate_min_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @field_validator("h3_resolutions")
    @classmethod
    def validate_resolutions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("At least one H3 resolution is required")
        if len(value) != len(set(value)) or any(
            item < 0 or item > 15 for item in value
        ):
            raise ValueError("H3 resolutions must be unique values within [0, 15]")
        return value

    @field_validator("frequencies")
    @classmethod
    def validate_frequencies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("At least one aggregation frequency is required")
        return value

    @model_validator(mode="after")
    def validate_sources_and_ranges(self) -> "SightingsPipelineConfig":
        if self.max_date is not None and date.fromisoformat(
            self.max_date
        ) < date.fromisoformat(self.min_date):
            raise ValueError("max_date cannot precede min_date")
        required_sources = {"twm", "acartia", "maplify", "inaturalist", "cwr", "gbif"}
        missing_sources = required_sources - set(self.collection.sources)
        if missing_sources:
            raise ValueError(f"Missing source configuration: {sorted(missing_sources)}")
        missing_chunks = set(self.frequencies) - set(self.aggregation_chunk_periods)
        if missing_chunks:
            raise ValueError(
                f"Missing aggregation chunk sizes: {sorted(missing_chunks)}"
            )
        for name, source in self.collection.sources.items():
            if not source.enabled:
                continue
            if name != "gbif":
                if not source.source_license or source.source_use_class is None:
                    raise ValueError(
                        f"Enabled {name} source requires explicit source_license "
                        "and source_use_class"
                    )
                if (
                    source.source_use_class == "REDISTRIBUTABLE"
                    and source.source_license.strip().upper()
                    in {"UNKNOWN", "UNVERIFIED", "NONE"}
                ):
                    raise ValueError(
                        f"Enabled {name} source cannot be redistributable with an unknown license"
                    )
            may_redistribute = source.source_use_class == "REDISTRIBUTABLE" or (
                name == "gbif"
                and any(
                    item.use_class == "REDISTRIBUTABLE"
                    for item in source.dataset_allowlist
                )
            )
            if may_redistribute:
                evidence = {
                    "source_license": source.source_license,
                    "source_license_terms_url": source.source_license_terms_url,
                    "source_attribution": source.source_attribution,
                    "source_license_reviewed_at": source.source_license_reviewed_at,
                    "source_license_version": source.source_license_version,
                    "source_license_jurisdiction": source.source_license_jurisdiction,
                }
                missing_evidence = sorted(
                    key for key, value in evidence.items() if not value
                )
                if missing_evidence:
                    raise ValueError(
                        f"Enabled {name} redistributable source lacks license evidence: "
                        f"{missing_evidence}"
                    )
                try:
                    date.fromisoformat(str(source.source_license_reviewed_at))
                except ValueError as exc:
                    raise ValueError(
                        f"Enabled {name} source_license_reviewed_at must be an ISO date"
                    ) from exc
                if not str(source.source_license_terms_url).startswith(
                    ("https://", "http://")
                ):
                    raise ValueError(
                        f"Enabled {name} source_license_terms_url must be HTTP(S)"
                    )
            if (
                name in {"acartia", "maplify", "inaturalist", "gbif"}
                and source.url is None
            ):
                raise ValueError(f"Enabled {name} source requires url")
            if name == "gbif":
                if source.dataset_metadata_url is None:
                    raise ValueError(
                        "Enabled gbif source requires dataset_metadata_url"
                    )
                if source.taxon_key is None or source.checklist_key is None:
                    raise ValueError(
                        "Enabled gbif source requires taxon_key and checklist_key"
                    )
                if source.taxon_key != "74SZC":
                    raise ValueError(
                        "gbif taxon_key must be the configured Orcinus orca key 74SZC"
                    )
                if source.checklist_key != "7ddf754f-d193-4cc9-b351-99906754a03b":
                    raise ValueError(
                        "gbif checklist_key must be the configured Catalogue of Life"
                    )
                if source.basis_of_record != "HUMAN_OBSERVATION":
                    raise ValueError("gbif basis_of_record must be HUMAN_OBSERVATION")
                if source.occurrence_status != "PRESENT":
                    raise ValueError("gbif occurrence_status must be PRESENT")
                if not source.require_no_geospatial_issue:
                    raise ValueError("gbif requires hasGeospatialIssue=false")
                if source.max_coordinate_uncertainty_m != 5000.0:
                    raise ValueError(
                        "gbif coordinate uncertainty limit must be 5000 metres"
                    )
                if not source.dataset_allowlist:
                    raise ValueError(
                        "Enabled gbif source requires a non-empty dataset_allowlist"
                    )
                keys = [item.key for item in source.dataset_allowlist]
                if len(keys) != len(set(keys)):
                    raise ValueError("gbif dataset_allowlist contains duplicate keys")
                overlap = set(keys) & set(source.excluded_dataset_keys)
                if overlap:
                    raise ValueError(
                        f"gbif dataset keys cannot be both allowed and excluded: {sorted(overlap)}"
                    )
                if "50c9509d-22c7-4a22-a47d-8c48425ef4a7" not in set(
                    source.excluded_dataset_keys
                ):
                    raise ValueError(
                        "gbif must explicitly exclude the iNaturalist dataset"
                    )
            if name == "cwr":
                if source.archive_index_url is None:
                    raise ValueError("Enabled cwr source requires archive_index_url")
                if (
                    source.archive_year_url_template is None
                    or "{year}" not in source.archive_year_url_template
                ):
                    raise ValueError(
                        "Enabled cwr source requires archive_year_url_template containing {year}"
                    )
                if not source.archive_years or len(source.archive_years) != len(
                    set(source.archive_years)
                ):
                    raise ValueError("Enabled cwr source requires unique archive_years")
                if not source.atlist_api_root or not source.atlist_maps:
                    raise ValueError(
                        "Enabled cwr source requires atlist_api_root and atlist_maps"
                    )
                if set(source.archive_years) & set(source.atlist_maps):
                    raise ValueError(
                        "cwr archive_years and atlist_maps must not overlap"
                    )
                map_ids = [item.map_id for item in source.atlist_maps.values()]
                if len(map_ids) != len(set(map_ids)):
                    raise ValueError("cwr atlist map IDs must be unique")
                if source.coordinate_bounds is None:
                    raise ValueError("Enabled cwr source requires coordinate_bounds")
                if source.source_use_class != "INTERNAL_ONLY":
                    raise ValueError(
                        "cwr must remain INTERNAL_ONLY until permission is confirmed"
                    )
                if not source.source_license:
                    raise ValueError(
                        "Enabled cwr source requires an explicit license status"
                    )
            if name == "inaturalist" and source.max_coordinate_uncertainty_m != 5000.0:
                raise ValueError(
                    "inaturalist coordinate uncertainty limit must be 5000 metres"
                )
        acartia = self.collection.sources.get("acartia")
        if (
            acartia is not None
            and acartia.enabled
            and not acartia.created_is_event_time
        ):
            raise ValueError(
                "Enabled Acartia source requires created_is_event_time=true"
            )
        missing_universes = {"SRKW", "TRANSIENT"} - set(self.model_universes)
        if missing_universes:
            raise ValueError(
                f"Missing model universe configuration: {sorted(missing_universes)}"
            )
        return self


def load_sightings_config(
    path: str | Path | ConfigDocument, *, workspace_root: str | Path | None = None
) -> tuple[ConfigDocument, SightingsPipelineConfig]:
    from .catalog import register_builtin_datasets

    register_builtin_datasets()
    document = (
        path
        if isinstance(path, ConfigDocument)
        else ConfigDocument.load(path, workspace_root=workspace_root)
    )
    settings = document.validate_as(SightingsPipelineConfig)
    # Older schema-v6 documents predate the explicit provider taxon parameter.
    source = settings.collection.sources["inaturalist"]
    if source.taxon_id is None:
        settings = settings.model_copy(
            update={
                "collection": settings.collection.model_copy(
                    update={
                        "sources": {
                            **settings.collection.sources,
                            "inaturalist": source.model_copy(
                                update={"taxon_id": 41521}
                            ),
                        }
                    }
                )
            }
        )
    return document, settings
