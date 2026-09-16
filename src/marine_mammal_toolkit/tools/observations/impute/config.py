from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Regime = Literal["retrospective", "operational_eod"]


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for date-based spatiotemporal context features.

    All temporal features use integer calendar-day lag. The canonical noon-UTC
    timestamp is never interpreted as a precise encounter time.
    """

    max_radius_km: float = 60.0
    distance_scale_km: float = 10.0
    time_scale_days: float = 4.0
    max_day_lag: int = 30
    support_unit_km: float = 2.5
    query_chunk_size: int = 2_000
    water_network_config_path: str = "config/data/environment_seascape.yaml"
    marine_h3_resolution: Literal[6, 8] = 6
    marine_fallback_to_haversine: bool = False
    max_target_snap_km: float = 5.0
    strong_local_support_floor: float = 0.01
    movement_speed_scale_km_per_day: float = 50.0
    maximum_feasible_km_per_day: float = 150.0
    known_other_labels: tuple[str, ...] = ("NRKW", "OFFSHORE", "OTHER")
    unknown_labels: tuple[str, ...] = ("UNKNOWN",)
    mixed_labels: tuple[str, ...] = ("MIXED",)
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

    def validate(self) -> None:
        if self.max_radius_km <= 0:
            raise ValueError("max_radius_km must be positive")
        if self.distance_scale_km <= 0:
            raise ValueError("distance_scale_km must be positive")
        if self.time_scale_days <= 0:
            raise ValueError("time_scale_days must be positive")
        if self.max_day_lag < 0:
            raise ValueError("max_day_lag must be nonnegative")
        if self.support_unit_km <= 0:
            raise ValueError("support_unit_km must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.max_target_snap_km < 0:
            raise ValueError("max_target_snap_km must be nonnegative")
        if self.strong_local_support_floor < 0:
            raise ValueError("strong_local_support_floor must be nonnegative")
        if self.movement_speed_scale_km_per_day <= 0:
            raise ValueError("movement_speed_scale_km_per_day must be positive")
        if self.maximum_feasible_km_per_day <= 0:
            raise ValueError("maximum_feasible_km_per_day must be positive")
        if self.marine_h3_resolution not in {6, 8}:
            raise ValueError("marine_h3_resolution must be 6 or 8")
        names = [item[0] for item in self.temporal_windows]
        if len(names) != len(set(names)):
            raise ValueError("temporal window names must be unique")
        for name, lo, hi in self.temporal_windows:
            if lo > hi:
                raise ValueError(f"invalid temporal window {name}: {lo} > {hi}")
            if abs(lo) > self.max_day_lag or abs(hi) > self.max_day_lag:
                raise ValueError(
                    f"temporal window {name} exceeds max_day_lag={self.max_day_lag}"
                )


@dataclass(frozen=True)
class ModelConfig:
    """Model, calibration, abstention, and validation settings."""

    random_state: int = 42
    n_splits: int = 3
    model_kind: Literal["extra_trees", "hist_gradient_boosting", "logistic"] = (
        "extra_trees"
    )
    n_estimators: int = 400
    max_features: float = 0.70
    learning_rate: float = 0.05
    max_iter: int = 350
    max_leaf_nodes: int = 23
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    logistic_c: float = 0.5
    calibration_method: Literal["auto", "sigmoid", "isotonic"] = "auto"
    conformal_alpha: float = 0.10
    target_selective_error: float = 0.05
    risk_confidence: float = 0.95
    min_policy_examples: int = 20
    min_policy_accepts: int = 12
    nested_decision_splits: int = 2
    ood_contamination: float = 0.01
    other_support_veto_ratio: float = 1.05
    other_support_veto_floor: float = 0.20
    blocked_days: int = 7
    blocked_km: float = 20.0
    minimum_labeled_per_class: int = 30
    max_evaluation_per_class: int | None = None
    ood_n_estimators: int = 100
    ood_min_regime_examples: int = 50
    max_training_majority_ratio: float = 10.0
    encounter_radius_km: float = 3.0
    encounter_association_radius_km: float = 20.0
    encounter_timestamp_tolerance_hours: float = 6.0
    use_regime_experts: bool = True
    min_expert_examples: int = 100
    enable_hyperparameter_tuning: bool = True
    tuning_max_candidates: int = 8
    tuning_n_jobs: int = 2
    enable_prediction_stability: bool = True
    maximum_stability_probability_shift: float = 0.20
    hard_certification_min_accepts_per_class: int = 100
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

    def validate(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.model_kind == "extra_trees" and self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.max_features <= 1:
            raise ValueError("max_features must be in (0, 1]")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal_alpha must be in (0, 1)")
        if not 0 < self.target_selective_error < 0.5:
            raise ValueError("target_selective_error must be in (0, 0.5)")
        if not 0.5 < self.risk_confidence < 1:
            raise ValueError("risk_confidence must be in (0.5, 1)")
        if self.nested_decision_splits != 2:
            raise ValueError(
                "nested_decision_splits must be 2 to reserve half of each outer fold"
            )
        if not 0 <= self.ood_contamination < 0.5:
            raise ValueError("ood_contamination must be in [0, 0.5)")
        if self.blocked_days <= 0 or self.blocked_km <= 0:
            raise ValueError("blocked_days and blocked_km must be positive")
        if (
            self.max_evaluation_per_class is not None
            and self.max_evaluation_per_class < 2
        ):
            raise ValueError("max_evaluation_per_class must be at least 2")
        if self.ood_n_estimators <= 0:
            raise ValueError("ood_n_estimators must be positive")
        if self.ood_min_regime_examples <= 0:
            raise ValueError("ood_min_regime_examples must be positive")
        if self.max_training_majority_ratio < 1:
            raise ValueError("max_training_majority_ratio must be >= 1")
        if self.encounter_radius_km <= 0:
            raise ValueError("encounter_radius_km must be positive")
        if self.encounter_association_radius_km < self.encounter_radius_km:
            raise ValueError("encounter association radius must be >= encounter radius")
        if self.encounter_timestamp_tolerance_hours <= 0:
            raise ValueError("encounter timestamp tolerance must be positive")
        if self.min_expert_examples < 20:
            raise ValueError("min_expert_examples must be at least 20")
        if self.tuning_max_candidates <= 0 or self.tuning_n_jobs <= 0:
            raise ValueError("tuning candidate and job counts must be positive")
        if not 0 < self.maximum_stability_probability_shift < 0.5:
            raise ValueError("maximum stability probability shift must be in (0, 0.5)")
        if self.hard_certification_min_accepts_per_class < 100:
            raise ValueError(
                "hard certification requires at least 100 accepts per class"
            )
        if self.final_calibration_strategy not in {
            "reconstruction",
            "encounter",
            "blocked",
            "purged_blocked",
        }:
            raise ValueError("invalid final_calibration_strategy")
        if self.hard_label_certification_strategy != "purged_blocked":
            raise ValueError(
                "hard-label certification must use purged_blocked evaluation"
            )


@dataclass(frozen=True)
class ImputationConfig:
    """Top-level configuration for the standalone imputation workflow."""

    regime: Regime = "retrospective"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    query_labels: tuple[str, ...] = ("UNKNOWN",)
    training_labels: tuple[str, ...] = ("SRKW", "TRANSIENT")
    map_year: int = 2025
    map_frame: Literal["day", "week", "month"] = "week"

    def validate(self) -> None:
        if self.regime not in {"retrospective", "operational_eod"}:
            raise ValueError(f"unsupported regime: {self.regime}")
        if set(self.training_labels) != {"SRKW", "TRANSIENT"}:
            raise ValueError("this package currently models SRKW versus TRANSIENT")
        self.feature.validate()
        self.model.validate()

    def to_dict(self) -> dict:
        return asdict(self)
