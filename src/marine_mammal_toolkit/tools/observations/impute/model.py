from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marine_mammal_toolkit.tools.observations.impute.calibration import (
    ClassConditionalConformal,
)
from marine_mammal_toolkit.tools.observations.impute.calibration import (
    ProbabilityCalibrator,
)
from marine_mammal_toolkit.tools.observations.impute.calibration import (
    crossfit_calibration,
)
from marine_mammal_toolkit.tools.observations.impute.calibration import (
    crossfit_conformal,
)
from marine_mammal_toolkit.tools.observations.impute.certification import (
    certification_release_ready,
)
from marine_mammal_toolkit.tools.observations.impute.certification import (
    validate_soft_count_certification,
)
from marine_mammal_toolkit.tools.observations.impute.config import ImputationConfig
from marine_mammal_toolkit.tools.observations.impute.encounters import (
    attach_encounter_ids,
)
from .components import FeatureBatch, ImputationComponents
from marine_mammal_toolkit.tools.observations.impute.io import add_anchor_quality
from marine_mammal_toolkit.tools.observations.impute.metrics import metrics_by_regime
from marine_mammal_toolkit.tools.observations.impute.metrics import reliability_table
from marine_mammal_toolkit.tools.observations.impute.metrics import risk_coverage_curve
from marine_mammal_toolkit.tools.observations.impute.metrics import (
    summarize_binary_metrics,
)
from marine_mammal_toolkit.tools.observations.impute.policy import AcceptancePolicy
from marine_mammal_toolkit.tools.observations.impute.policy import crossfit_policy
from marine_mammal_toolkit.tools.observations.impute.splits import iter_splits
from marine_mammal_toolkit.tools.observations.impute.splits import (
    make_spatiotemporal_groups,
)
from marine_mammal_toolkit.tools.observations.impute.splits import (
    purge_spatiotemporal_neighbors,
)

LABEL_TO_INT = {"TRANSIENT": 0, "SRKW": 1}
INT_TO_LABEL = {0: "TRANSIENT", 1: "SRKW"}
MODEL_VERSION = "date_context_selective_v8_binary_diagnostic_only"
DIAGNOSTIC_META_COLUMNS = (
    "NEAREST_EVIDENCE_ENCOUNTER_ID",
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
    "DISTANCE_METHOD",
    "MARINE_CANDIDATE_NEIGHBORS",
    "MARINE_ROUTED_NEIGHBORS",
    "MARINE_FALLBACK_NEIGHBORS",
    "MARINE_BARRIER_EXCLUDED_NEIGHBORS",
    "MARINE_OUT_OF_RANGE_NEIGHBORS",
    "MARINE_LOOKUP_COVERAGE",
    "MARINE_TARGET_KNOWN",
    "MARINE_TARGET_SNAP_KM",
)


def mark_encounter_label_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag every member of an encounter containing contradictory ecotype evidence."""

    if "ENCOUNTER_ID" not in frame or "ECOTYPE_DETAIL" not in frame:
        raise ValueError(
            "Encounter conflict detection requires ENCOUNTER_ID and ECOTYPE_DETAIL"
        )
    out = frame.copy()
    encounter = out["ENCOUNTER_ID"].astype(str)
    detail = out["ECOTYPE_DETAIL"].fillna("UNKNOWN").astype(str).str.upper()
    binary = detail.where(detail.isin(LABEL_TO_INT))
    mixed_binary = binary.groupby(encounter, sort=False).transform("nunique").ge(2)
    row_conflict = detail.eq("MIXED")
    if "LABEL_CONFLICT" in out:
        row_conflict = row_conflict | out["LABEL_CONFLICT"].fillna(False).astype(bool)
    out["ENCOUNTER_LABEL_CONFLICT"] = (
        row_conflict.groupby(encounter, sort=False).transform("any") | mixed_binary
    )
    return out


def certified_soft_mass_mask(
    *,
    policy_accepted: np.ndarray,
    locally_supported: np.ndarray,
    binary_domain_supported: np.ndarray,
    stability_evaluated: np.ndarray,
    prediction_stable: np.ndarray,
    encounter_conflict: np.ndarray,
    certified_stratum_supported: np.ndarray,
    open_set_probability_mass_validated: np.ndarray,
    certification: dict[str, Any] | None,
) -> np.ndarray:
    """Return the fail-closed eligibility mask for probabilistic count mass."""

    arrays = [
        np.asarray(value, dtype=bool)
        for value in (
            policy_accepted,
            locally_supported,
            binary_domain_supported,
            stability_evaluated,
            prediction_stable,
            encounter_conflict,
            certified_stratum_supported,
            open_set_probability_mass_validated,
        )
    ]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("Soft-mass eligibility inputs must have identical lengths")
    certified = certification_release_ready(certification)
    return (
        certified
        & arrays[0]
        & arrays[1]
        & arrays[2]
        & arrays[3]
        & arrays[4]
        & ~arrays[5]
        & arrays[6]
        & arrays[7]
    )


def soft_certification_stratum_mask(
    frame: pd.DataFrame,
    certification: dict[str, Any] | None,
) -> np.ndarray:
    """Match rows only to explicitly certified source/era selectors."""

    supported = np.zeros(len(frame), dtype=bool)
    if not certification_release_ready(certification) or frame.empty:
        return supported
    if "SOURCE" not in frame or "SIGHTING_DATE" not in frame:
        return supported
    sources = frame["SOURCE"].fillna("").astype(str).str.upper()
    dates = pd.to_datetime(frame["SIGHTING_DATE"], errors="coerce").dt.normalize()
    for selector in certification.get("ELIGIBLE_STRATA", []):
        if not isinstance(selector, dict):
            continue
        start = pd.Timestamp(selector.get("era_start"))
        end = pd.Timestamp(selector.get("era_end"))
        supported |= (
            sources.eq(str(selector.get("source") or "").upper())
            & dates.ge(start)
            & dates.le(end)
        ).to_numpy(dtype=bool)
    return supported


def validate_imputation_mass(
    frame: pd.DataFrame,
    *,
    query_labels: tuple[str, ...] = ("UNKNOWN",),
) -> None:
    """Assert finite probabilities and the four-class unit-mass contract."""

    if frame.empty:
        return
    required = {
        "P_SRKW",
        "P_TRANSIENT",
        "P_OTHER",
        "EXPECTED_SRKW_COUNT",
        "EXPECTED_TRANSIENT_COUNT",
        "EXPECTED_OTHER_COUNT",
        "EXPECTED_UNKNOWN_COUNT",
        "USE_FOR_HARD_COUNTS",
        "USE_FOR_PROBABILISTIC_COUNTS",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Imputation mass validation is missing columns: {missing}")

    detail_column = (
        "ECOTYPE_DETAIL_OBSERVED"
        if "ECOTYPE_DETAIL_OBSERVED" in frame
        else "ECOTYPE_DETAIL"
    )
    detail = frame[detail_column].fillna("UNKNOWN").astype(str).str.upper()
    query = detail.isin(query_labels)
    binary_observed = detail.isin({"SRKW", "TRANSIENT"})
    modeled_binary = binary_observed | query
    probability = frame[["P_SRKW", "P_TRANSIENT"]].apply(pd.to_numeric, errors="coerce")
    scored = probability.notna().any(axis=1)
    if probability.loc[scored].isna().any().any():
        raise ValueError("Imputation probabilities must be present as a pair")
    scored_values = probability.loc[scored].to_numpy(dtype=float)
    if not np.isfinite(scored_values).all():
        raise ValueError("Imputation probabilities must be finite")
    if ((scored_values < -1e-9) | (scored_values > 1 + 1e-9)).any():
        raise ValueError("Imputation probabilities must lie within [0, 1]")
    if not np.allclose(scored_values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("SRKW and transient probabilities must have unit mass")
    if not scored.loc[modeled_binary].all():
        raise ValueError(
            "Every modeled observation requires finite binary probabilities"
        )
    p_other = pd.to_numeric(frame["P_OTHER"], errors="coerce")
    known_other = ~modeled_binary
    if not np.allclose(p_other.loc[known_other].to_numpy(dtype=float), 1.0, atol=1e-8):
        raise ValueError("Known-OTHER observations must have P_OTHER=1")
    if not np.allclose(
        p_other.loc[binary_observed].to_numpy(dtype=float), 0.0, atol=1e-8
    ):
        raise ValueError("Observed binary labels must have P_OTHER=0")
    query_other = p_other.loc[query].dropna().to_numpy(dtype=float)
    if len(query_other) and (
        not np.isfinite(query_other).all()
        or ((query_other < -1e-9) | (query_other > 1 + 1e-9)).any()
    ):
        raise ValueError(
            "Query P_OTHER diagnostics must be finite values within [0, 1]"
        )

    expected = frame[
        [
            "EXPECTED_SRKW_COUNT",
            "EXPECTED_TRANSIENT_COUNT",
            "EXPECTED_OTHER_COUNT",
            "EXPECTED_UNKNOWN_COUNT",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    expected_values = expected.to_numpy(dtype=float)
    if not np.isfinite(expected_values).all():
        raise ValueError("Expected imputation count mass must be finite")
    if ((expected_values < -1e-9) | (expected_values > 1 + 1e-9)).any():
        raise ValueError("Expected imputation count mass must lie within [0, 1]")
    expected_sum = expected_values.sum(axis=1)
    if not np.allclose(expected_sum, 1.0, atol=1e-8):
        raise ValueError(
            "Every observation must retain exactly one unit of expected mass"
        )

    hard = frame["USE_FOR_HARD_COUNTS"].fillna(False).astype(bool).to_numpy()
    probabilistic = (
        frame["USE_FOR_PROBABILISTIC_COUNTS"].fillna(False).astype(bool).to_numpy()
    )
    if np.any(hard & ~probabilistic):
        raise ValueError(
            "Hard-count eligibility must imply probabilistic-count eligibility"
        )
    query_mask = query.to_numpy()
    srkw_observed = detail.eq("SRKW").to_numpy()
    transient_observed = detail.eq("TRANSIENT").to_numpy()
    known_other_mask = known_other.to_numpy()
    expected_srkw = expected["EXPECTED_SRKW_COUNT"].to_numpy(dtype=float)
    expected_transient = expected["EXPECTED_TRANSIENT_COUNT"].to_numpy(dtype=float)
    expected_other = expected["EXPECTED_OTHER_COUNT"].to_numpy(dtype=float)
    expected_unknown = expected["EXPECTED_UNKNOWN_COUNT"].to_numpy(dtype=float)
    if not (
        np.allclose(expected_srkw[srkw_observed], 1.0, atol=1e-8)
        and np.allclose(expected_transient[srkw_observed], 0.0, atol=1e-8)
        and np.allclose(expected_other[srkw_observed], 0.0, atol=1e-8)
        and np.allclose(expected_unknown[srkw_observed], 0.0, atol=1e-8)
    ):
        raise ValueError("Observed SRKW rows must retain unit SRKW expected mass")
    if not (
        np.allclose(expected_srkw[transient_observed], 0.0, atol=1e-8)
        and np.allclose(expected_transient[transient_observed], 1.0, atol=1e-8)
        and np.allclose(expected_other[transient_observed], 0.0, atol=1e-8)
        and np.allclose(expected_unknown[transient_observed], 0.0, atol=1e-8)
    ):
        raise ValueError(
            "Observed transient rows must retain unit transient expected mass"
        )
    if not (
        np.allclose(expected_srkw[known_other_mask], 0.0, atol=1e-8)
        and np.allclose(expected_transient[known_other_mask], 0.0, atol=1e-8)
        and np.allclose(expected_other[known_other_mask], 1.0, atol=1e-8)
        and np.allclose(expected_unknown[known_other_mask], 0.0, atol=1e-8)
    ):
        raise ValueError("Known-OTHER rows must retain unit OTHER expected mass")
    if not np.allclose(expected_unknown[query_mask & probabilistic], 0.0, atol=1e-8):
        raise ValueError("Count-eligible queries must allocate no mass to UNKNOWN")
    if not np.allclose(expected_unknown[query_mask & ~probabilistic], 1.0, atol=1e-8):
        raise ValueError("Count-ineligible queries must retain all mass as UNKNOWN")
    counted_query = query_mask & probabilistic
    probability_srkw = probability["P_SRKW"].to_numpy(dtype=float)
    probability_transient = probability["P_TRANSIENT"].to_numpy(dtype=float)
    if not (
        np.allclose(
            expected_srkw[counted_query], probability_srkw[counted_query], atol=1e-8
        )
        and np.allclose(
            expected_transient[counted_query],
            probability_transient[counted_query],
            atol=1e-8,
        )
        and np.allclose(expected_other[counted_query], 0.0, atol=1e-8)
    ):
        raise ValueError(
            "Count-eligible query mass must match calibrated binary probabilities"
        )
    abstained_query = query_mask & ~probabilistic
    if not (
        np.allclose(expected_srkw[abstained_query], 0.0, atol=1e-8)
        and np.allclose(expected_transient[abstained_query], 0.0, atol=1e-8)
        and np.allclose(expected_other[abstained_query], 0.0, atol=1e-8)
    ):
        raise ValueError(
            "Count-ineligible queries must not leak attributed expected mass"
        )
    stability_evaluated = (
        (
            frame.get("STABILITY_EVALUATED", False)
            if isinstance(frame.get("STABILITY_EVALUATED", False), pd.Series)
            else pd.Series(False, index=frame.index)
        )
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    if np.any(query_mask & probabilistic & ~stability_evaluated):
        raise ValueError(
            "Every count-eligible query requires an evaluated stability check"
        )
    soft_count_certified = (
        (
            frame.get("SOFT_COUNT_CERTIFIED", False)
            if isinstance(frame.get("SOFT_COUNT_CERTIFIED", False), pd.Series)
            else pd.Series(False, index=frame.index)
        )
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    soft_counted = query_mask & probabilistic & ~hard
    if np.any(soft_counted & ~soft_count_certified):
        raise ValueError(
            "Soft probabilistic count mass requires explicit certification"
        )


@dataclass
class EvaluationResult:
    strategy: str
    metrics: dict[str, Any]
    oof_predictions: pd.DataFrame
    reliability: pd.DataFrame
    risk_coverage: pd.DataFrame
    metrics_by_regime: pd.DataFrame
    calibrator: ProbabilityCalibrator
    conformal: ClassConditionalConformal
    policy: AcceptancePolicy


def certify_hard_label_classes(
    evaluations: dict[str, EvaluationResult],
    *,
    strategy: str,
    target_selective_error: float,
    minimum_accepted_per_class: int = 100,
    release_enabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """Certify hard labels exclusively from the configured leakage-safe arm."""

    if strategy not in evaluations:
        raise ValueError(
            f"hard_label_certification_strategy={strategy!r} was not evaluated; "
            f"available={sorted(evaluations)}"
        )
    metrics = evaluations[strategy].metrics
    class_risk = metrics["ENCOUNTER_LEVEL"]["PREDICTED_CLASS_RISK"]
    promotion_gate = metrics.get("HARD_LABEL_PROMOTION_GATE", {})
    leakage_gate = promotion_gate.get("ZERO_LINEAGE_LEAKAGE") is True
    strata_gate = promotion_gate.get("REQUIRED_STRATA_PASSED") is True
    certification: dict[str, dict[str, Any]] = {}
    for class_name in ("TRANSIENT", "SRKW"):
        risk = class_risk[class_name]
        accepted_n = int(risk["accepted_n"])
        risk_gate = float(risk["error_upper"]) <= target_selective_error
        certification[class_name] = {
            "CERTIFIED": bool(
                release_enabled
                and accepted_n >= minimum_accepted_per_class
                and risk_gate
                and leakage_gate
                and strata_gate
            ),
            "OUTER_ACCEPTED_N": accepted_n,
            "OUTER_ERRORS": int(risk["errors"]),
            "OUTER_ERROR_UPPER": float(risk["error_upper"]),
            "STRATEGY": strategy,
            "MIN_ACCEPTED_REQUIRED": int(minimum_accepted_per_class),
            "RISK_GATE_PASSED": bool(risk_gate),
            "ZERO_LINEAGE_LEAKAGE": bool(leakage_gate),
            "REQUIRED_STRATA_PASSED": bool(strata_gate),
            "RELEASE_ENABLED": bool(release_enabled),
        }
    return certification


@dataclass
class _RegimeMixture:
    fallback: Any
    experts: dict[str, Any]


class _OODDetector:
    def __init__(
        self,
        contamination: float,
        random_state: int,
        n_estimators: int = 100,
        min_regime_examples: int = 50,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.min_regime_examples = min_regime_examples
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold_: float = -np.inf
        self.regime_thresholds_: dict[str, float] = {}

    def fit(
        self, X: pd.DataFrame, regimes: np.ndarray | pd.Series | None = None
    ) -> "_OODDetector":
        transformed = self.imputer.fit_transform(X)
        transformed = self.scaler.fit_transform(transformed)
        self.model.fit(transformed)
        scores = self.model.score_samples(transformed)
        self.threshold_ = (
            float(np.quantile(scores, self.contamination))
            if self.contamination > 0
            else -np.inf
        )
        self.regime_thresholds_ = {}
        if regimes is not None and self.contamination > 0:
            regime_array = np.asarray(regimes, dtype=str)
            for regime in np.unique(regime_array):
                mask = regime_array == regime
                if int(mask.sum()) >= self.min_regime_examples:
                    self.regime_thresholds_[str(regime)] = float(
                        np.quantile(scores[mask], self.contamination)
                    )
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.imputer.transform(X)
        transformed = self.scaler.transform(transformed)
        return self.model.score_samples(transformed)

    def thresholds(self, regimes: np.ndarray | pd.Series | None, n: int) -> np.ndarray:
        threshold = np.full(n, self.threshold_, dtype=float)
        if regimes is not None:
            regime_array = np.asarray(regimes, dtype=str)
            for regime, value in self.regime_thresholds_.items():
                threshold[regime_array == regime] = value
        return threshold

    def inlier(
        self, X: pd.DataFrame, regimes: np.ndarray | pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores = self.score(X)
        thresholds = self.thresholds(regimes, len(scores))
        return scores >= thresholds, scores, thresholds


class SelectiveDateContextImputer:
    """Probabilistic SRKW/transient attribution with calibrated abstention.

    The primary regime is retrospective. An optional operational_eod regime uses
    same-day and earlier-day context only. Both regimes use calendar-day lags;
    noon UTC is treated as a date label rather than a precise event time.
    """

    def __init__(
        self,
        config: ImputationConfig | None = None,
        *,
        components: ImputationComponents,
    ):
        self.components = components
        self.config = config or ImputationConfig()
        self.config.validate()
        self.feature_builder = self.components.feature_builder(self.config.feature)
        self.model_: Any | None = None
        self.calibrator_: ProbabilityCalibrator | None = None
        self.conformal_: ClassConditionalConformal | None = None
        self.policy_: AcceptancePolicy | None = None
        self.ood_: _OODDetector | None = None
        self.feature_names_: list[str] = []
        self.anchors_: pd.DataFrame | None = None
        self.evaluations_: dict[str, EvaluationResult] = {}
        self.training_summary_: dict[str, Any] = {}
        self.tuned_params_: dict[str, Any] = {}
        self.tuning_results_: pd.DataFrame = pd.DataFrame()
        self.encounter_lookup_: pd.DataFrame = pd.DataFrame()
        self.class_certification_: dict[str, dict[str, Any]] = {}
        self.soft_mass_certification_: dict[str, Any] = {
            "CERTIFIED": False,
            "EVIDENCE_ID": None,
        }
        self.model_domain_geometries_: dict[str, Any] = {}
        self.model_domain_provenance_: dict[str, Any] = {}
        self.encounter_quarantine_summary_: dict[str, int] = {}

    def set_soft_mass_certification(
        self,
        *,
        certified: bool,
        evidence: dict[str, Any] | str | Path | None = None,
    ) -> "SelectiveDateContextImputer":
        """Certify soft count mass only from complete, validated evidence."""

        if certified:
            if evidence is None:
                raise ValueError(
                    "Enabling soft count mass requires certification evidence"
                )
            validate_soft_count_certification(evidence)
            raise NotImplementedError(
                "This binary imputer cannot release soft counts; a typed, checksum-bound "
                "open-set scorer with unconditional SRKW/Transient/Other mass is required"
            )
        else:
            self.soft_mass_certification_ = {
                "CERTIFIED": False,
                "EVIDENCE_ID": None,
                "EVIDENCE_SHA256": None,
                "SCHEMA_VERSION": None,
            }
        if self.training_summary_:
            self.training_summary_["SOFT_COUNT_CERTIFICATION"] = dict(
                self.soft_mass_certification_
            )
        return self

    def set_model_domains(
        self,
        geometries: dict[str, Any],
        *,
        provenance: dict[str, Any] | None = None,
    ) -> "SelectiveDateContextImputer":
        """Attach reviewed ecotype model domains; no geometric fallback is allowed."""

        normalized = {
            str(name).upper(): geometry for name, geometry in geometries.items()
        }
        missing = sorted({"SRKW", "TRANSIENT"} - set(normalized))
        if missing:
            raise ValueError(f"Imputation model domains are missing: {missing}")
        for name in ("SRKW", "TRANSIENT"):
            geometry = normalized[name]
            if geometry is None or bool(getattr(geometry, "is_empty", True)):
                raise ValueError(f"Imputation model domain is empty: {name}")
        self.model_domain_geometries_ = {
            name: normalized[name] for name in ("SRKW", "TRANSIENT")
        }
        self.model_domain_provenance_ = dict(provenance or {})
        if self.training_summary_:
            self.training_summary_["MODEL_DOMAIN_STATUS"] = "CONFIGURED"
            self.training_summary_["MODEL_DOMAIN_PROVENANCE"] = dict(
                self.model_domain_provenance_
            )
        return self

    def _domain_support(self, query: pd.DataFrame) -> dict[str, np.ndarray]:
        domains = getattr(self, "model_domain_geometries_", {}) or {}
        points = shapely.points(
            query["LONGITUDE"].to_numpy(dtype=float),
            query["LATITUDE"].to_numpy(dtype=float),
        )
        result: dict[str, np.ndarray] = {}
        for name in ("SRKW", "TRANSIENT"):
            geometry = domains.get(name)
            result[name] = (
                np.asarray(shapely.covers(geometry, points), dtype=bool)
                if geometry is not None
                else np.zeros(len(query), dtype=bool)
            )
        return result

    def _new_model(
        self,
        seed_offset: int = 0,
        *,
        overrides: dict[str, Any] | None = None,
        n_jobs: int = -1,
    ):
        cfg = self.config.model
        seed = cfg.random_state + seed_offset
        params = {**self.tuned_params_, **(overrides or {})}
        if cfg.model_kind == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                learning_rate=cfg.learning_rate,
                max_iter=cfg.max_iter,
                max_leaf_nodes=cfg.max_leaf_nodes,
                min_samples_leaf=cfg.min_samples_leaf,
                l2_regularization=cfg.l2_regularization,
                class_weight="balanced",
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=seed,
            )
        if cfg.model_kind == "extra_trees":
            return Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "classifier",
                        ExtraTreesClassifier(
                            n_estimators=int(
                                params.get("n_estimators", cfg.n_estimators)
                            ),
                            max_features=float(
                                params.get("max_features", cfg.max_features)
                            ),
                            min_samples_leaf=int(
                                params.get("min_samples_leaf", cfg.min_samples_leaf)
                            ),
                            max_depth=params.get("max_depth"),
                            class_weight="balanced",
                            n_jobs=n_jobs,
                            random_state=seed,
                        ),
                    ),
                ]
            )
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=cfg.logistic_c,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _fit_model(
        model: Any, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray
    ) -> Any:
        if isinstance(model, Pipeline):
            model.fit(X, y, classifier__sample_weight=weights)
        else:
            model.fit(X, y, sample_weight=weights)
        return model

    @staticmethod
    def _predict_base(model: Any, X: pd.DataFrame) -> np.ndarray:
        probability = model.predict_proba(X)
        classes = list(
            model.classes_ if not isinstance(model, Pipeline) else model[-1].classes_
        )
        srkw_index = classes.index(1)
        return np.asarray(probability[:, srkw_index], dtype=float)

    @staticmethod
    def _expert_key(regime: str) -> str:
        if regime.startswith("SAME_DAY"):
            return "SAME_DAY"
        if regime.startswith("LAGGED") or regime == "WEAK_LOCAL":
            return "LAGGED"
        return "GLOBAL"

    def _fit_predictor(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        regimes: np.ndarray | pd.Series,
        *,
        seed_offset: int = 0,
    ) -> Any:
        fallback = self._new_model(seed_offset=seed_offset)
        self._fit_model(fallback, X, y, weights)
        if not self.config.model.use_regime_experts:
            return fallback
        regime_array = np.asarray(regimes, dtype=str)
        keys = np.asarray([self._expert_key(item) for item in regime_array], dtype=str)
        experts: dict[str, Any] = {}
        for expert_offset, key in enumerate(("SAME_DAY", "LAGGED"), start=1):
            mask = keys == key
            if int(mask.sum()) < self.config.model.min_expert_examples:
                continue
            if np.unique(y[mask]).size < 2:
                continue
            expert = self._new_model(seed_offset=seed_offset + 100 * expert_offset)
            self._fit_model(expert, X.loc[mask], y[mask], weights[mask])
            experts[key] = expert
        return _RegimeMixture(fallback=fallback, experts=experts)

    def _predict_raw(
        self, model: Any, X: pd.DataFrame, regimes: np.ndarray | pd.Series
    ) -> np.ndarray:
        if not isinstance(model, _RegimeMixture):
            return self._predict_base(model, X)
        regime_array = np.asarray(regimes, dtype=str)
        keys = np.asarray([self._expert_key(item) for item in regime_array], dtype=str)
        probability = self._predict_base(model.fallback, X)
        for key, expert in model.experts.items():
            mask = keys == key
            if mask.any():
                probability[mask] = self._predict_base(expert, X.loc[mask])
        return probability

    @staticmethod
    def _align_features(batch: FeatureBatch, names: list[str] | None) -> FeatureBatch:
        if names is None:
            return batch
        return FeatureBatch(X=batch.X.reindex(columns=names), meta=batch.meta)

    @staticmethod
    def _encounter_representative_positions(frame: pd.DataFrame) -> np.ndarray:
        """Choose one deterministic calibration unit per encounter and class."""

        if frame.empty:
            return np.asarray([], dtype=int)
        units = frame.reset_index(drop=True).copy()
        units["_POSITION"] = np.arange(len(units), dtype=int)
        if "ENCOUNTER_ID" not in units:
            units["ENCOUNTER_ID"] = units["OBSERVATION_ID"].astype(str)
        if "ANCHOR_WEIGHT" not in units:
            units["ANCHOR_WEIGHT"] = 1.0
        representatives = (
            units.sort_values(
                ["ANCHOR_WEIGHT", "OBSERVATION_ID"], ascending=[False, True]
            )
            .drop_duplicates(["ENCOUNTER_ID", "MODEL_CLASS"], keep="first")["_POSITION"]
            .to_numpy(dtype=int)
        )
        return np.sort(representatives)

    def _sample_evaluation_rows(
        self, known: pd.DataFrame, cap_per_class: int
    ) -> pd.DataFrame:
        """Maximize encounter coverage before adding duplicate reports."""

        pieces: list[pd.DataFrame] = []
        for class_offset, (_, group) in enumerate(
            known.groupby("MODEL_CLASS", sort=False)
        ):
            if len(group) <= cap_per_class:
                pieces.append(group)
                continue
            representatives = self._encounter_representative_positions(group)
            representative_rows = group.iloc[representatives]
            if len(representative_rows) >= cap_per_class:
                pieces.append(
                    representative_rows.sample(
                        n=cap_per_class,
                        random_state=self.config.model.random_state + class_offset,
                    )
                )
                continue
            remaining = group.drop(index=representative_rows.index)
            extra_n = cap_per_class - len(representative_rows)
            extras = remaining.sample(
                n=extra_n,
                random_state=self.config.model.random_state + 100 + class_offset,
            )
            pieces.append(pd.concat([representative_rows, extras]))
        return pd.concat(pieces, ignore_index=True)

    def _classifier_training_sample(
        self,
        frame: pd.DataFrame,
        *,
        seed_offset: int = 0,
    ) -> pd.DataFrame:
        """Deduplicate encounters and cap the majority class for fitting.

        This sampling applies only to classifier rows. Every eligible known
        sighting remains available in the anchor table that creates contextual
        evidence features.
        """

        if frame.empty:
            return frame.copy()
        sample = frame.copy()
        if "ENCOUNTER_ID" not in sample:
            sample["ENCOUNTER_ID"] = sample["OBSERVATION_ID"].astype(str)
        sample = sample.sort_values(
            ["ANCHOR_WEIGHT", "OBSERVATION_ID"], ascending=[False, True]
        )
        sample = sample.drop_duplicates(["ENCOUNTER_ID", "MODEL_CLASS"], keep="first")

        counts = sample["MODEL_CLASS"].value_counts()
        if len(counts) < 2:
            raise ValueError("Classifier training sample must contain both classes")
        minority_n = int(counts.min())
        majority_cap = max(
            minority_n,
            int(np.ceil(minority_n * self.config.model.max_training_majority_ratio)),
        )
        pieces = []
        for class_name, group in sample.groupby("MODEL_CLASS", sort=False):
            cap = majority_cap if len(group) > minority_n else len(group)
            pieces.append(
                group.sample(
                    n=min(len(group), cap),
                    random_state=self.config.model.random_state + seed_offset,
                )
            )
        return (
            pd.concat(pieces, ignore_index=True)
            .sort_values("OBSERVATION_ID")
            .reset_index(drop=True)
        )

    def _eligible_frames(
        self,
        observations: pd.DataFrame,
        associations: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config.model
        frame = attach_encounter_ids(
            observations,
            associations,
            radius_km=cfg.encounter_radius_km,
            association_radius_km=cfg.encounter_association_radius_km,
            timestamp_tolerance_hours=cfg.encounter_timestamp_tolerance_hours,
        )
        frame = add_anchor_quality(frame, associations)
        frame = mark_encounter_label_conflicts(frame)
        self.encounter_lookup_ = frame[
            [
                "OBSERVATION_ID",
                "ENCOUNTER_ID",
                "ENCOUNTER_SIZE",
                "ENCOUNTER_LABEL_CONFLICT",
            ]
        ].copy()
        frame = self.components.prepare_frame(frame, self.config.feature)
        quarantined = frame["ENCOUNTER_LABEL_CONFLICT"].fillna(False).astype(bool)
        self.encounter_quarantine_summary_ = {
            "MIXED_ECOTYPE_ENCOUNTER_N": int(
                frame.loc[quarantined, "ENCOUNTER_ID"].nunique()
            ),
            "MIXED_ECOTYPE_QUARANTINED_ROW_N": int(quarantined.sum()),
            "MIXED_ECOTYPE_QUARANTINED_LABELED_N": int(
                (quarantined & frame["MODEL_CLASS"].isin(["SRKW", "TRANSIENT"])).sum()
            ),
            "MIXED_ECOTYPE_QUARANTINED_OTHER_N": int(
                (quarantined & frame["MODEL_CLASS"].eq("OTHER")).sum()
            ),
        }
        known = frame.loc[
            frame["MODEL_CLASS"].isin(["SRKW", "TRANSIENT"]) & ~quarantined
        ].copy()
        known = known.loc[~known.get("LABEL_CONFLICT", False)].reset_index(drop=True)
        other = frame.loc[frame["MODEL_CLASS"].eq("OTHER") & ~quarantined].copy()
        other = other.reset_index(drop=True)
        query = frame.loc[frame["ECOTYPE_DETAIL"].isin(self.config.query_labels)].copy()
        query = query.reset_index(drop=True)
        counts = known["MODEL_CLASS"].value_counts()
        minimum = self.config.model.minimum_labeled_per_class
        for cls in ("SRKW", "TRANSIENT"):
            if int(counts.get(cls, 0)) < minimum:
                raise ValueError(
                    f"Need at least {minimum} labeled {cls} observations; found {counts.get(cls, 0)}"
                )
        return known, other, query

    def _tune_hyperparameters(self, known: pd.DataFrame, other: pd.DataFrame) -> None:
        cfg = self.config.model
        if not cfg.enable_hyperparameter_tuning or cfg.model_kind != "extra_trees":
            self.tuning_results_ = pd.DataFrame()
            self.tuned_params_ = {}
            return
        y_all = known["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
        groups = known["ENCOUNTER_ID"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(
            n_splits=min(4, len(np.unique(groups))),
            shuffle=True,
            random_state=cfg.random_state + 40_000,
        )
        tuning_splits = list(splitter.split(known, y_all, groups))
        selected_split = next(
            (
                (train_idx, validation_idx)
                for train_idx, validation_idx in tuning_splits
                if np.unique(y_all[train_idx]).size == 2
                and np.unique(y_all[validation_idx]).size == 2
            ),
            None,
        )
        if selected_split is None:
            raise ValueError(
                "Tuning needs an encounter-held-out split with both classes"
            )
        _, validation_idx = selected_split
        validation_encounters = set(
            known.iloc[validation_idx]["ENCOUNTER_ID"].astype(str)
        )
        allowed_known = known.loc[
            ~known["ENCOUNTER_ID"].astype(str).isin(validation_encounters)
        ].copy()
        validation = self._classifier_training_sample(
            known.iloc[validation_idx].copy(), seed_offset=40_000
        )
        training = self._classifier_training_sample(allowed_known, seed_offset=40_001)
        anchors = pd.concat([allowed_known, other], ignore_index=True)
        train_batch = self.feature_builder.transform(
            training, anchors, regime=self.config.regime, exclude_self=True
        )
        validation_batch = self._align_features(
            self.feature_builder.transform(
                validation, anchors, regime=self.config.regime, exclude_self=True
            ),
            list(train_batch.X.columns),
        )
        train_y = training["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
        validation_y = validation["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
        weights = training["ANCHOR_WEIGHT"].to_numpy(dtype=float)
        candidate_grid = [
            {"min_samples_leaf": leaf, "max_features": features, "max_depth": depth}
            for leaf, features, depth in (
                (10, 0.50, None),
                (10, 0.75, None),
                (20, 0.50, None),
                (20, 0.75, None),
                (35, 0.60, None),
                (35, 0.85, 24),
                (50, 0.70, None),
                (20, 1.00, 24),
            )
        ][: cfg.tuning_max_candidates]

        def score_candidate(index: int, params: dict[str, Any]) -> dict[str, Any]:
            model = self._new_model(
                seed_offset=50_000 + index,
                overrides=params,
                n_jobs=1,
            )
            self._fit_model(model, train_batch.X, train_y, weights)
            probability = self._predict_base(model, validation_batch.X)
            return {
                **params,
                "BRIER": float(brier_score_loss(validation_y, probability)),
                "LOG_LOSS": float(log_loss(validation_y, probability, labels=[0, 1])),
                "ROC_AUC": (
                    float(roc_auc_score(validation_y, probability))
                    if np.unique(validation_y).size == 2
                    else np.nan
                ),
                "TRAIN_N": len(training),
                "VALIDATION_N": len(validation),
            }

        rows = Parallel(n_jobs=cfg.tuning_n_jobs, prefer="threads")(
            delayed(score_candidate)(index, params)
            for index, params in enumerate(candidate_grid)
        )
        results = (
            pd.DataFrame(rows)
            .sort_values(
                ["BRIER", "LOG_LOSS", "ROC_AUC"], ascending=[True, True, False]
            )
            .reset_index(drop=True)
        )
        winner = results.iloc[0]
        self.tuned_params_ = {
            "min_samples_leaf": int(winner["min_samples_leaf"]),
            "max_features": float(winner["max_features"]),
            "max_depth": (
                None if pd.isna(winner["max_depth"]) else int(winner["max_depth"])
            ),
        }
        self.tuning_results_ = results

    def _evaluate_strategy(
        self,
        known: pd.DataFrame,
        other: pd.DataFrame,
        *,
        strategy: str,
        full_known: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        full_known = known if full_known is None else full_known.reset_index(drop=True)
        y = known["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
        groups = (
            known["ENCOUNTER_ID"].astype(str)
            if strategy == "encounter"
            else (
                make_spatiotemporal_groups(
                    known,
                    block_days=self.config.model.blocked_days,
                    block_km=self.config.model.blocked_km,
                )
                if strategy in {"blocked", "purged_blocked"}
                else None
            )
        )
        splits = list(
            iter_splits(
                known,
                y,
                strategy=strategy,
                n_splits=self.config.model.n_splits,
                random_state=self.config.model.random_state,
                groups=groups,
            )
        )
        full_groups = (
            full_known["ENCOUNTER_ID"].astype(str)
            if strategy == "encounter"
            else (
                make_spatiotemporal_groups(
                    full_known,
                    block_days=self.config.model.blocked_days,
                    block_km=self.config.model.blocked_km,
                )
                if strategy in {"blocked", "purged_blocked"}
                else None
            )
        )
        if strategy == "purged_blocked":
            purged_splits = []
            for train_idx, test_idx in splits:
                purged_train = purge_spatiotemporal_neighbors(
                    known,
                    train_idx,
                    test_idx,
                    purge_days=self.config.feature.max_day_lag,
                    purge_km=self.config.feature.max_radius_km,
                )
                if np.unique(y[purged_train]).size < 2:
                    raise ValueError(
                        "Purged blocked fold lost a training class; reduce the purge "
                        "window or increase the evaluation sample"
                    )
                purged_splits.append((purged_train, test_idx))
            splits = purged_splits

        raw = np.full(len(known), np.nan, dtype=float)
        ood_score = np.full(len(known), np.nan, dtype=float)
        ood_threshold = np.full(len(known), np.nan, dtype=float)
        ood_inlier = np.zeros(len(known), dtype=bool)
        meta_frames: list[pd.DataFrame] = []
        feature_names: list[str] | None = None
        calibrated = np.full(len(y), np.nan, dtype=float)
        singleton = np.full(len(y), -1, dtype=int)
        set_size = np.full(len(y), -1, dtype=int)
        pvalues = np.full((len(y), 2), np.nan, dtype=float)
        accepted = np.zeros(len(y), dtype=bool)
        threshold = np.full(len(y), np.inf, dtype=float)
        reason = np.full(len(y), "NOT_EVALUATED", dtype=object)

        for fold_index, (train_idx, test_idx) in enumerate(splits):
            test = known.iloc[test_idx].copy()
            test_ids = set(test["OBSERVATION_ID"].astype(str))
            if strategy == "reconstruction":
                allowed_known = full_known.loc[
                    ~full_known["OBSERVATION_ID"].astype(str).isin(test_ids)
                ].copy()
            elif strategy in {"encounter", "blocked"}:
                test_group_values = set(np.asarray(groups)[test_idx])
                allowed_known = full_known.loc[
                    ~pd.Series(np.asarray(full_groups), index=full_known.index).isin(
                        test_group_values
                    )
                ].copy()
            else:
                combined = pd.concat([full_known, test], ignore_index=True)
                allowed_positions = purge_spatiotemporal_neighbors(
                    combined,
                    np.arange(len(full_known), dtype=int),
                    np.arange(len(full_known), len(combined), dtype=int),
                    purge_days=self.config.feature.max_day_lag,
                    purge_km=self.config.feature.max_radius_km,
                )
                allowed_known = full_known.iloc[allowed_positions].copy()

            # Spatial blocks and row-level reconstruction must never leave a
            # sibling report from the held-out encounter in training/context.
            test_encounters = set(test["ENCOUNTER_ID"].astype(str))
            allowed_known = allowed_known.loc[
                ~allowed_known["ENCOUNTER_ID"].astype(str).isin(test_encounters)
            ].copy()

            # Reserve encounter groups from the outer training fold exclusively
            # for calibration, conformal scores, and acceptance-policy fitting.
            # Neither their labels nor the outer test labels enter the classifier.
            outer_train = known.iloc[train_idx].copy()
            outer_train = outer_train.loc[
                ~outer_train["ENCOUNTER_ID"].astype(str).isin(test_encounters)
            ].reset_index(drop=True)
            outer_y = outer_train["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
            outer_groups = outer_train["ENCOUNTER_ID"].astype(str).to_numpy()
            inner_split_count = min(
                self.config.model.nested_decision_splits,
                len(np.unique(outer_groups)),
            )
            if inner_split_count < 2:
                raise ValueError(
                    "Nested decision calibration needs at least two encounters"
                )
            inner_splitter = StratifiedGroupKFold(
                n_splits=inner_split_count,
                shuffle=True,
                random_state=self.config.model.random_state + 20_000 + fold_index,
            )
            inner_splits = list(
                inner_splitter.split(outer_train, outer_y, outer_groups)
            )
            selected_inner = next(
                (
                    calibration_idx
                    for classifier_idx, calibration_idx in inner_splits
                    if np.unique(outer_y[classifier_idx]).size == 2
                    and np.unique(outer_y[calibration_idx]).size == 2
                ),
                None,
            )
            if selected_inner is None:
                raise ValueError(
                    "Nested calibration needs an encounter split containing both classes"
                )
            calibration_local_idx = selected_inner
            calibration = outer_train.iloc[calibration_local_idx].copy()
            if calibration["MODEL_CLASS"].nunique() < 2:
                raise ValueError("Nested calibration fold must contain both classes")
            calibration_encounters = set(calibration["ENCOUNTER_ID"].astype(str))
            allowed_known = allowed_known.loc[
                ~allowed_known["ENCOUNTER_ID"].astype(str).isin(calibration_encounters)
            ].copy()

            train = self._classifier_training_sample(
                allowed_known, seed_offset=fold_index
            )
            anchors = pd.concat([allowed_known, other], ignore_index=True)
            train_batch = self.feature_builder.transform(
                train,
                anchors,
                regime=self.config.regime,
                exclude_self=True,
            )
            if feature_names is None:
                feature_names = list(train_batch.X.columns)
            train_batch = self._align_features(train_batch, feature_names)
            test_batch = self._align_features(
                self.feature_builder.transform(
                    test,
                    anchors,
                    regime=self.config.regime,
                    exclude_self=True,
                ),
                feature_names,
            )
            calibration_batch = self._align_features(
                self.feature_builder.transform(
                    calibration,
                    anchors,
                    regime=self.config.regime,
                    exclude_self=True,
                ),
                feature_names,
            )
            fold_y = train["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
            if len(np.unique(fold_y)) < 2:
                raise ValueError(f"Fold {fold_index} has only one training class")
            weights = train["ANCHOR_WEIGHT"].to_numpy(dtype=float)
            model = self._fit_predictor(
                train_batch.X,
                fold_y,
                weights,
                train_batch.meta["EVIDENCE_REGIME"],
                seed_offset=fold_index,
            )
            raw[test_idx] = self._predict_raw(
                model, test_batch.X, test_batch.meta["EVIDENCE_REGIME"]
            )
            calibration_raw = self._predict_raw(
                model,
                calibration_batch.X,
                calibration_batch.meta["EVIDENCE_REGIME"],
            )

            detector = _OODDetector(
                contamination=self.config.model.ood_contamination,
                random_state=self.config.model.random_state + fold_index,
                n_estimators=self.config.model.ood_n_estimators,
                min_regime_examples=self.config.model.ood_min_regime_examples,
            ).fit(train_batch.X, train_batch.meta["EVIDENCE_REGIME"])
            inlier, scores, fold_ood_threshold = detector.inlier(
                test_batch.X, test_batch.meta["EVIDENCE_REGIME"]
            )
            ood_inlier[test_idx] = inlier
            ood_score[test_idx] = scores
            ood_threshold[test_idx] = fold_ood_threshold

            calibration_ood, _, _ = detector.inlier(
                calibration_batch.X, calibration_batch.meta["EVIDENCE_REGIME"]
            )
            calibration_y = (
                calibration["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
            )
            calibration_representatives = self._encounter_representative_positions(
                calibration
            )
            calibration_fit_y = calibration_y[calibration_representatives]
            if np.unique(calibration_fit_y).size < 2:
                raise ValueError(
                    "Encounter-balanced nested calibration must contain both classes"
                )
            fold_calibrator = ProbabilityCalibrator(
                method=self.config.model.calibration_method,
                random_state=self.config.model.random_state + 30_000 + fold_index,
            ).fit(
                calibration_raw[calibration_representatives],
                calibration_fit_y,
            )
            calibration_probability = fold_calibrator.predict(calibration_raw)
            test_probability = fold_calibrator.predict(raw[test_idx])
            fold_conformal = ClassConditionalConformal(
                alpha=self.config.model.conformal_alpha
            ).fit(
                calibration_probability[calibration_representatives],
                calibration_fit_y,
            )
            calibration_singleton, _, _ = fold_conformal.prediction_sets(
                calibration_probability
            )
            test_singleton, test_size, test_pvalues = fold_conformal.prediction_sets(
                test_probability
            )
            calibration_veto = self.components.support_veto(
                calibration_batch.meta,
                ratio=self.config.model.other_support_veto_ratio,
                floor=self.config.model.other_support_veto_floor,
            )
            test_veto = self.components.support_veto(
                test_batch.meta,
                ratio=self.config.model.other_support_veto_ratio,
                floor=self.config.model.other_support_veto_floor,
            )
            fold_policy = self.components.acceptance_policy(
                target_error=self.config.model.target_selective_error,
                confidence=self.config.model.risk_confidence,
                min_examples=self.config.model.min_policy_examples,
                min_accepts=self.config.model.min_policy_accepts,
                hard_abstain_regimes=self.config.model.hard_abstain_regimes,
            ).fit(
                calibration_fit_y,
                calibration_probability[calibration_representatives],
                calibration_singleton[calibration_representatives],
                calibration_ood[calibration_representatives],
                calibration_batch.meta["EVIDENCE_REGIME"].to_numpy(dtype=str)[
                    calibration_representatives
                ],
                calibration_veto[calibration_representatives],
            )
            fold_accepted, fold_threshold, fold_reason = fold_policy.apply(
                test_probability,
                test_singleton,
                inlier,
                test_batch.meta["EVIDENCE_REGIME"].to_numpy(dtype=str),
                test_veto,
            )
            calibrated[test_idx] = test_probability
            singleton[test_idx] = test_singleton
            set_size[test_idx] = test_size
            pvalues[test_idx] = test_pvalues
            accepted[test_idx] = fold_accepted
            threshold[test_idx] = fold_threshold
            reason[test_idx] = fold_reason

            fold_meta = test_batch.meta.copy()
            fold_meta["_POSITION"] = test_idx
            meta_frames.append(fold_meta)

        if (
            np.isnan(raw).any()
            or np.isnan(ood_score).any()
            or np.isnan(ood_threshold).any()
        ):
            raise RuntimeError(f"OOF evaluation for {strategy} did not cover all rows")
        meta = pd.concat(meta_frames).sort_values("_POSITION")
        if not np.array_equal(meta["_POSITION"].to_numpy(), np.arange(len(known))):
            raise RuntimeError("OOF metadata order is inconsistent")
        meta = meta.drop(columns="_POSITION").reset_index(drop=True)

        regimes = meta["EVIDENCE_REGIME"].to_numpy(dtype=str)
        veto = self.components.support_veto(
            meta,
            ratio=self.config.model.other_support_veto_ratio,
            floor=self.config.model.other_support_veto_floor,
        )

        if (
            np.isnan(calibrated).any()
            or np.isnan(pvalues).any()
            or (set_size < 0).any()
        ):
            raise RuntimeError("Decision-stack cross-fitting did not cover every row")

        # Fit production calibration/conformal/policy objects from cross-fitted raw
        # probabilities across the full labeled corpus. These objects are later
        # applied to the model refitted on all known labels.
        decision_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for train_idx, test_idx in splits:
            local_representatives = self._encounter_representative_positions(
                known.iloc[train_idx]
            )
            decision_splits.append(
                (np.asarray(train_idx)[local_representatives], test_idx)
            )
        final_representatives = self._encounter_representative_positions(known)

        calibrated_for_final, _ = crossfit_calibration(
            raw,
            y,
            decision_splits,
            method=self.config.model.calibration_method,
            random_state=self.config.model.random_state,
        )
        final_calibrator = ProbabilityCalibrator(
            method=self.config.model.calibration_method,
            random_state=self.config.model.random_state,
        ).fit(raw[final_representatives], y[final_representatives])
        final_singleton, _, _, _ = crossfit_conformal(
            calibrated_for_final,
            y,
            decision_splits,
            alpha=self.config.model.conformal_alpha,
        )
        final_conformal = ClassConditionalConformal(
            alpha=self.config.model.conformal_alpha
        ).fit(
            calibrated_for_final[final_representatives],
            y[final_representatives],
        )
        _, _, _, _ = crossfit_policy(
            y,
            calibrated_for_final,
            final_singleton,
            ood_inlier,
            regimes,
            veto,
            decision_splits,
            target_error=self.config.model.target_selective_error,
            confidence=self.config.model.risk_confidence,
            min_examples=self.config.model.min_policy_examples,
            min_accepts=self.config.model.min_policy_accepts,
            hard_abstain_regimes=self.config.model.hard_abstain_regimes,
        )
        final_policy = self.components.acceptance_policy(
            target_error=self.config.model.target_selective_error,
            confidence=self.config.model.risk_confidence,
            min_examples=self.config.model.min_policy_examples,
            min_accepts=self.config.model.min_policy_accepts,
            hard_abstain_regimes=self.config.model.hard_abstain_regimes,
        ).fit(
            y[final_representatives],
            calibrated_for_final[final_representatives],
            final_singleton[final_representatives],
            ood_inlier[final_representatives],
            regimes[final_representatives],
            veto[final_representatives],
        )

        metrics: dict[str, Any] = summarize_binary_metrics(
            y,
            calibrated,
            accepted=accepted,
            confidence=self.config.model.risk_confidence,
        )
        metrics["RAW_PROBABILITY_METRICS"] = summarize_binary_metrics(
            y,
            raw,
            accepted=np.ones(len(y), dtype=bool),
            confidence=self.config.model.risk_confidence,
        )
        metrics["STRATEGY"] = strategy
        metrics["CONFORMAL_ALPHA"] = self.config.model.conformal_alpha
        metrics["CONFORMAL_SINGLETON_RATE"] = float((set_size == 1).mean())
        metrics["OOD_INLIER_RATE"] = float(ood_inlier.mean())
        metrics["OTHER_VETO_RATE"] = float(veto.mean())
        metrics["TRAIN_N_MIN"] = int(min(len(train_idx) for train_idx, _ in splits))
        metrics["TRAIN_N_MAX"] = int(max(len(train_idx) for train_idx, _ in splits))
        encounter_representatives = self._encounter_representative_positions(known)
        metrics["ENCOUNTER_LEVEL"] = summarize_binary_metrics(
            y[encounter_representatives],
            calibrated[encounter_representatives],
            accepted=accepted[encounter_representatives],
            confidence=self.config.model.risk_confidence,
        )
        metrics["ENCOUNTER_LEVEL"]["UNIT_N"] = int(len(encounter_representatives))

        predicted = (calibrated >= 0.5).astype(int)
        prediction_set = np.where(
            set_size == 0,
            "{}",
            np.where(
                set_size == 2,
                "{TRANSIENT,SRKW}",
                np.where(singleton == 1, "{SRKW}", "{TRANSIENT}"),
            ),
        )
        oof_columns = [
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "SIGHTING_DATE",
            "LATITUDE",
            "LONGITUDE",
            "MODEL_CLASS",
            *[
                column
                for column in (
                    "SOURCE",
                    "OBSERVATION_QUALITY_TIER",
                    "SOURCE_TIME_PRECISION",
                )
                if column in known
            ],
        ]
        oof = known[oof_columns].copy()
        years = pd.to_datetime(oof["SIGHTING_DATE"], errors="coerce").dt.year
        era_start = (years // 5) * 5
        oof["ERA"] = np.where(
            years.notna(),
            era_start.astype("Int64").astype(str)
            + "-"
            + (era_start + 4).astype("Int64").astype(str),
            "UNKNOWN",
        )
        oof["SPATIAL_REGION"] = (
            pd.to_numeric(oof["LATITUDE"], errors="coerce")
            .round()
            .astype("Int64")
            .astype(str)
            + ":"
            + pd.to_numeric(oof["LONGITUDE"], errors="coerce")
            .round()
            .astype("Int64")
            .astype(str)
        )
        oof["Y_TRUE"] = y
        oof["P_SRKW_RAW"] = raw
        oof["P_SRKW"] = calibrated
        oof["P_TRANSIENT"] = 1 - calibrated
        oof["PREDICTED_CLASS"] = [INT_TO_LABEL[item] for item in predicted]
        oof["CONFORMAL_SET"] = prediction_set
        oof["CONFORMAL_SET_SIZE"] = set_size
        oof["CONFORMAL_P_TRANSIENT"] = pvalues[:, 0]
        oof["CONFORMAL_P_SRKW"] = pvalues[:, 1]
        oof["OOD_SCORE"] = ood_score
        oof["OOD_THRESHOLD"] = ood_threshold
        oof["OOD_MARGIN"] = ood_score - ood_threshold
        oof["OOD_INLIER"] = ood_inlier
        oof["OTHER_SUPPORT_VETO"] = veto
        oof["EVIDENCE_REGIME"] = meta["EVIDENCE_REGIME"].to_numpy()
        oof["ACCEPTED"] = accepted
        oof["ACCEPTANCE_THRESHOLD"] = threshold
        oof["DECISION_REASON"] = reason
        representative_mask = np.zeros(len(known), dtype=bool)
        representative_mask[encounter_representatives] = True
        oof["IS_ENCOUNTER_REPRESENTATIVE"] = representative_mask
        for column in DIAGNOSTIC_META_COLUMNS:
            oof[column] = meta[column].to_numpy()

        eligible_for_curve = (set_size == 1) & ood_inlier & ~veto
        return EvaluationResult(
            strategy=strategy,
            metrics=metrics,
            oof_predictions=oof,
            reliability=reliability_table(y, calibrated),
            risk_coverage=risk_coverage_curve(
                y,
                calibrated,
                eligible=eligible_for_curve,
                confidence=self.config.model.risk_confidence,
            ),
            metrics_by_regime=metrics_by_regime(
                y,
                calibrated,
                accepted,
                meta["EVIDENCE_REGIME"].to_numpy(dtype=str),
                confidence=self.config.model.risk_confidence,
            ),
            calibrator=final_calibrator,
            conformal=final_conformal,
            policy=final_policy,
        )

    def fit(
        self,
        observations: pd.DataFrame,
        associations: pd.DataFrame | None = None,
        *,
        evaluate_strategies: tuple[str, ...] = (
            "reconstruction",
            "encounter",
            "blocked",
        ),
    ) -> "SelectiveDateContextImputer":
        known, other, query = self._eligible_frames(observations, associations)
        self._tune_hyperparameters(known, other)
        evaluation_known = known
        evaluation_cap = self.config.model.max_evaluation_per_class
        if evaluation_cap is not None:
            evaluation_known = self._sample_evaluation_rows(known, evaluation_cap)
        evaluations: dict[str, EvaluationResult] = {}
        for strategy in evaluate_strategies:
            evaluations[strategy] = self._evaluate_strategy(
                evaluation_known,
                other,
                strategy=strategy,
                full_known=known,
            )
        selected = self.config.model.final_calibration_strategy
        certification_strategy = self.config.model.hard_label_certification_strategy
        if selected not in evaluations:
            raise ValueError(
                f"final_calibration_strategy={selected!r} was not evaluated; "
                f"available={sorted(evaluations)}"
            )
        self.class_certification_ = certify_hard_label_classes(
            evaluations,
            strategy=certification_strategy,
            target_selective_error=self.config.model.target_selective_error,
            minimum_accepted_per_class=(
                self.config.model.hard_certification_min_accepts_per_class
            ),
            release_enabled=self.config.model.enable_hard_label_release,
        )

        anchors = pd.concat([known, other], ignore_index=True)
        classifier_train = self._classifier_training_sample(known)
        train_batch = self.feature_builder.transform(
            classifier_train,
            anchors,
            regime=self.config.regime,
            exclude_self=True,
        )
        self.feature_names_ = list(train_batch.X.columns)
        y = classifier_train["MODEL_CLASS"].map(LABEL_TO_INT).to_numpy(dtype=int)
        self.model_ = self._fit_predictor(
            train_batch.X,
            y,
            classifier_train["ANCHOR_WEIGHT"].to_numpy(dtype=float),
            train_batch.meta["EVIDENCE_REGIME"],
        )
        self.ood_ = _OODDetector(
            contamination=self.config.model.ood_contamination,
            random_state=self.config.model.random_state,
            n_estimators=self.config.model.ood_n_estimators,
            min_regime_examples=self.config.model.ood_min_regime_examples,
        ).fit(train_batch.X, train_batch.meta["EVIDENCE_REGIME"])
        self.calibrator_ = evaluations[selected].calibrator
        self.conformal_ = evaluations[selected].conformal
        self.policy_ = evaluations[selected].policy
        self.anchors_ = anchors.copy()
        self.evaluations_ = evaluations
        self.training_summary_ = {
            "MODEL_VERSION": MODEL_VERSION,
            "REGIME": self.config.regime,
            "LABELED_N": int(len(known)),
            "EVALUATION_LABELED_N": int(len(evaluation_known)),
            "CLASSIFIER_TRAINING_N": int(len(classifier_train)),
            "LABELED_ENCOUNTER_N": int(known["ENCOUNTER_ID"].nunique()),
            "CLASSIFIER_ENCOUNTER_N": int(classifier_train["ENCOUNTER_ID"].nunique()),
            "SRKW_N": int((known["MODEL_CLASS"] == "SRKW").sum()),
            "TRANSIENT_N": int((known["MODEL_CLASS"] == "TRANSIENT").sum()),
            "KNOWN_OTHER_ANCHORS_N": int(len(other)),
            "UNKNOWN_QUERY_N": int(len(query)),
            "FEATURE_N": int(len(self.feature_names_)),
            "FINAL_CALIBRATION_STRATEGY": selected,
            "HARD_LABEL_CERTIFICATION_STRATEGY": certification_strategy,
            "TUNED_PARAMETERS": self.tuned_params_,
            "REGIME_EXPERTS": (
                sorted(self.model_.experts)
                if isinstance(self.model_, _RegimeMixture)
                else []
            ),
            "CLASS_CERTIFICATION": self.class_certification_,
            "SOFT_COUNT_CERTIFICATION": dict(
                getattr(
                    self,
                    "soft_mass_certification_",
                    {"CERTIFIED": False, "EVIDENCE_ID": None},
                )
            ),
            "MODEL_DOMAIN_STATUS": (
                "CONFIGURED"
                if getattr(self, "model_domain_geometries_", {})
                else "NOT_CONFIGURED"
            ),
            "MODEL_DOMAIN_PROVENANCE": dict(
                getattr(self, "model_domain_provenance_", {})
            ),
            "OPEN_SET_STATUS": "DEFERRED_NO_GOLD_OTHER_LABELS",
            **self.encounter_quarantine_summary_,
        }
        return self

    def _check_fitted(self) -> None:
        if any(
            item is None
            for item in (
                self.model_,
                self.calibrator_,
                self.conformal_,
                self.policy_,
                self.ood_,
                self.anchors_,
            )
        ):
            raise RuntimeError("The imputer has not been fitted")

    def _prediction_stability(
        self,
        query: pd.DataFrame,
        base_batch: FeatureBatch,
        base_probability: np.ndarray,
        policy_accepted: np.ndarray,
        *,
        enabled: bool,
    ) -> dict[str, np.ndarray]:
        """Stress accepted predictions by removing influential evidence."""

        n = len(query)
        result = {
            "P_SRKW_NO_SAME_DAY": np.full(n, np.nan, dtype=float),
            "P_SRKW_NO_FUTURE": np.full(n, np.nan, dtype=float),
            "P_SRKW_NO_NEAREST_ENCOUNTER": np.full(n, np.nan, dtype=float),
            "MAX_STABILITY_PROBABILITY_SHIFT": np.full(n, np.nan, dtype=float),
            "STABILITY_CLASS_AGREEMENT": np.zeros(n, dtype=bool),
            "PREDICTION_STABLE": np.zeros(n, dtype=bool),
            "STABILITY_EVALUATED": np.zeros(n, dtype=bool),
        }
        candidate_positions = np.flatnonzero(policy_accepted)
        if not enabled or len(candidate_positions) == 0:
            result["PREDICTION_STABLE"][candidate_positions] = True
            result["STABILITY_CLASS_AGREEMENT"][candidate_positions] = True
            return result

        for name in (
            "P_SRKW_NO_SAME_DAY",
            "P_SRKW_NO_FUTURE",
            "P_SRKW_NO_NEAREST_ENCOUNTER",
        ):
            result[name][candidate_positions] = base_probability[candidate_positions]

        def run_check(
            name: str,
            positions: np.ndarray,
            **transform_kwargs: Any,
        ) -> None:
            if len(positions) == 0:
                return
            counterfactual_batch = self.feature_builder.transform(
                query.iloc[positions].copy(),
                self.anchors_,  # type: ignore[arg-type]
                regime=self.config.regime,
                exclude_self=True,
                **transform_kwargs,
            )
            aligned = self._align_features(counterfactual_batch, self.feature_names_)
            raw = self._predict_raw(
                self.model_, aligned.X, aligned.meta["EVIDENCE_REGIME"]
            )
            probability = self.calibrator_.predict(raw)  # type: ignore[union-attr]
            result[name][positions] = probability

        same_support = (
            base_batch.meta.iloc[candidate_positions][
                [
                    "SAME_DAY_SRKW_SUPPORT",
                    "SAME_DAY_TRANSIENT_SUPPORT",
                    "SAME_DAY_OTHER_SUPPORT",
                ]
            ]
            .sum(axis=1)
            .to_numpy(dtype=float)
        )
        run_check(
            "P_SRKW_NO_SAME_DAY",
            candidate_positions[same_support > 0],
            exclude_same_day=True,
        )
        nearest_ids = base_batch.meta.iloc[candidate_positions][
            "NEAREST_EVIDENCE_ENCOUNTER_ID"
        ].to_numpy(dtype=object)
        nearest_present = ~pd.isna(nearest_ids)
        run_check(
            "P_SRKW_NO_NEAREST_ENCOUNTER",
            candidate_positions[nearest_present],
            excluded_anchor_encounters=nearest_ids[nearest_present],
        )
        if self.config.regime == "retrospective":
            future_columns = [
                column
                for column in base_batch.X.columns
                if column.startswith("future_") and column.endswith("__support")
            ]
            future_support = (
                base_batch.X.iloc[candidate_positions][future_columns]
                .sum(axis=1)
                .to_numpy(dtype=float)
            )
            run_check(
                "P_SRKW_NO_FUTURE",
                candidate_positions[future_support > 0],
                exclude_future=True,
            )

        alternate_probabilities = [
            result[name][candidate_positions]
            for name in (
                "P_SRKW_NO_SAME_DAY",
                "P_SRKW_NO_FUTURE",
                "P_SRKW_NO_NEAREST_ENCOUNTER",
            )
        ]
        alternatives = np.column_stack(alternate_probabilities)
        base = base_probability[candidate_positions]
        base_class = base >= 0.5
        class_agreement = np.all((alternatives >= 0.5) == base_class[:, None], axis=1)
        max_shift = np.max(np.abs(alternatives - base[:, None]), axis=1)
        stable = class_agreement & (
            max_shift <= self.config.model.maximum_stability_probability_shift
        )
        result["MAX_STABILITY_PROBABILITY_SHIFT"][candidate_positions] = max_shift
        result["STABILITY_CLASS_AGREEMENT"][candidate_positions] = class_agreement
        result["PREDICTION_STABLE"][candidate_positions] = stable
        result["STABILITY_EVALUATED"][candidate_positions] = True
        return result

    def predict_queries(
        self,
        observations: pd.DataFrame,
        *,
        compute_stability: bool | None = None,
    ) -> pd.DataFrame:
        self._check_fitted()
        enriched = observations.copy()
        if not self.encounter_lookup_.empty:
            lookup = self.encounter_lookup_.set_index("OBSERVATION_ID")
            identifiers = enriched["OBSERVATION_ID"].astype(str)
            enriched["ENCOUNTER_ID"] = identifiers.map(lookup["ENCOUNTER_ID"])
            enriched["ENCOUNTER_SIZE"] = identifiers.map(lookup["ENCOUNTER_SIZE"])
            enriched["ENCOUNTER_LABEL_CONFLICT"] = (
                identifiers.map(lookup["ENCOUNTER_LABEL_CONFLICT"])
                if "ENCOUNTER_LABEL_CONFLICT" in lookup
                else False
            )
            enriched["ENCOUNTER_ID"] = enriched["ENCOUNTER_ID"].fillna(identifiers)
            enriched["ENCOUNTER_SIZE"] = (
                enriched["ENCOUNTER_SIZE"].fillna(1).astype(int)
            )
            enriched["ENCOUNTER_LABEL_CONFLICT"] = (
                enriched["ENCOUNTER_LABEL_CONFLICT"].fillna(False).astype(bool)
            )
        elif "ENCOUNTER_LABEL_CONFLICT" not in enriched:
            enriched["ENCOUNTER_LABEL_CONFLICT"] = False
        frame = self.components.prepare_frame(enriched, self.config.feature)
        query = frame.loc[frame["ECOTYPE_DETAIL"].isin(self.config.query_labels)].copy()
        if query.empty:
            return query.assign(
                P_SRKW=pd.Series(dtype=float),
                P_TRANSIENT=pd.Series(dtype=float),
                IMPUTATION_APPLIED=pd.Series(dtype=bool),
            )
        batch = self.feature_builder.transform(
            query,
            self.anchors_,  # type: ignore[arg-type]
            regime=self.config.regime,
            exclude_self=True,
        )
        batch = self._align_features(batch, self.feature_names_)
        raw = self._predict_raw(self.model_, batch.X, batch.meta["EVIDENCE_REGIME"])
        calibrated = self.calibrator_.predict(raw)  # type: ignore[union-attr]
        singleton, set_size, pvalues = self.conformal_.prediction_sets(calibrated)  # type: ignore[union-attr]
        ood_inlier, ood_score, ood_threshold = self.ood_.inlier(  # type: ignore[union-attr]
            batch.X, batch.meta["EVIDENCE_REGIME"]
        )
        veto = self.components.support_veto(
            batch.meta,
            ratio=self.config.model.other_support_veto_ratio,
            floor=self.config.model.other_support_veto_floor,
        )
        policy_accepted, threshold, reason = self.policy_.apply(  # type: ignore[union-attr]
            calibrated,
            singleton,
            ood_inlier,
            batch.meta["EVIDENCE_REGIME"].to_numpy(dtype=str),
            veto,
        )
        predicted = (calibrated >= 0.5).astype(int)
        domain_support = self._domain_support(query)
        srkw_domain_supported = domain_support["SRKW"]
        transient_domain_supported = domain_support["TRANSIENT"]
        binary_domain_supported = srkw_domain_supported & transient_domain_supported
        predicted_domain_supported = np.where(
            predicted == 1, srkw_domain_supported, transient_domain_supported
        )
        encounter_conflict = (
            query["ENCOUNTER_LABEL_CONFLICT"].fillna(False).astype(bool).to_numpy()
        )
        class_certified = np.asarray(
            [
                bool(
                    getattr(self, "class_certification_", {})
                    .get(INT_TO_LABEL[item], {})
                    .get("CERTIFIED", False)
                )
                for item in predicted
            ],
            dtype=bool,
        )
        locally_supported = (
            batch.meta["EVIDENCE_REGIME"]
            .isin(
                [
                    "SAME_DAY",
                    "SAME_DAY_CONFLICT",
                    "LAGGED",
                    "LAGGED_CONFLICT",
                    "WEAK_LOCAL",
                ]
            )
            .to_numpy(dtype=bool)
            & (
                batch.meta["MARINE_ROUTED_NEIGHBORS"].to_numpy(dtype=float)
                + batch.meta["MARINE_FALLBACK_NEIGHBORS"].to_numpy(dtype=float)
                > 0
            )
            & ood_inlier
            & ~veto
        )
        hard_candidate = (
            policy_accepted
            & class_certified
            & predicted_domain_supported
            & ~encounter_conflict
        )
        soft_candidate = (
            policy_accepted
            & locally_supported
            & binary_domain_supported
            & ~encounter_conflict
        )
        stability_candidate = hard_candidate | soft_candidate
        stability = self._prediction_stability(
            query,
            batch,
            calibrated,
            stability_candidate,
            enabled=(
                self.config.model.enable_prediction_stability
                if compute_stability is None
                else compute_stability
            ),
        )
        stability_evaluated = stability["STABILITY_EVALUATED"]
        prediction_stable = stability["PREDICTION_STABLE"]
        accepted = hard_candidate & stability_evaluated & prediction_stable
        soft_mass_certification = getattr(
            self,
            "soft_mass_certification_",
            {"CERTIFIED": False, "EVIDENCE_ID": None},
        )
        certified_stratum_supported = soft_certification_stratum_mask(
            query, soft_mass_certification
        )
        soft_eligible = (
            certified_soft_mass_mask(
                policy_accepted=policy_accepted,
                locally_supported=locally_supported,
                binary_domain_supported=binary_domain_supported,
                stability_evaluated=stability_evaluated,
                prediction_stable=prediction_stable,
                encounter_conflict=encounter_conflict,
                certified_stratum_supported=certified_stratum_supported,
                # This model emits conditional binary probabilities. It has no
                # open-set Other mass and can never release fractional counts.
                open_set_probability_mass_validated=np.zeros(len(query), dtype=bool),
                certification=soft_mass_certification,
            )
            & ~accepted
        )
        stability_not_evaluated = stability_candidate & ~stability_evaluated
        unstable = stability_candidate & stability_evaluated & ~prediction_stable
        uncertified = policy_accepted & ~class_certified
        reason = np.asarray(reason, dtype=object)
        reason[uncertified] = "CLASS_NOT_OUTER_VALIDATED"
        reason[policy_accepted & ~predicted_domain_supported] = (
            "OUTSIDE_PREDICTED_CLASS_DOMAIN"
        )
        reason[uncertified & locally_supported & ~binary_domain_supported] = (
            "OUTSIDE_BINARY_MODEL_DOMAIN"
        )
        reason[stability_not_evaluated] = "STABILITY_NOT_EVALUATED"
        reason[unstable] = "PREDICTION_UNSTABLE"
        soft_not_certified = (
            soft_candidate
            & stability_evaluated
            & prediction_stable
            & ~certification_release_ready(soft_mass_certification)
        )
        reason[soft_not_certified] = "SOFT_MASS_NOT_CERTIFIED"
        soft_stratum_not_certified = (
            soft_candidate
            & stability_evaluated
            & prediction_stable
            & certification_release_ready(soft_mass_certification)
            & ~certified_stratum_supported
        )
        reason[soft_stratum_not_certified] = "SOFT_STRATUM_NOT_CERTIFIED"
        reason[encounter_conflict] = "MIXED_ECOTYPE_ENCOUNTER"
        imputed = np.where(
            accepted, np.where(predicted == 1, "SRKW", "TRANSIENT"), pd.NA
        )
        prediction_set = np.where(
            set_size == 0,
            "{}",
            np.where(
                set_size == 2,
                "{TRANSIENT,SRKW}",
                np.where(singleton == 1, "{SRKW}", "{TRANSIENT}"),
            ),
        )

        out = query.copy()
        out["ECOTYPE_DETAIL_OBSERVED"] = out["ECOTYPE_DETAIL"]
        out["P_SRKW_RAW"] = raw
        out["P_SRKW"] = calibrated
        out["P_TRANSIENT"] = 1 - calibrated
        # No open-set probability is fabricated without gold OTHER labels.
        out["P_OTHER"] = np.nan
        out["PREDICTED_CLASS"] = [INT_TO_LABEL[item] for item in predicted]
        out["CONFORMAL_SET"] = prediction_set
        out["CONFORMAL_SET_SIZE"] = set_size
        out["CONFORMAL_P_TRANSIENT"] = pvalues[:, 0]
        out["CONFORMAL_P_SRKW"] = pvalues[:, 1]
        out["OOD_SCORE"] = ood_score
        out["OOD_THRESHOLD"] = ood_threshold
        out["OOD_MARGIN"] = ood_score - ood_threshold
        out["OOD_INLIER"] = ood_inlier
        out["OTHER_SUPPORT_VETO"] = veto
        out["EVIDENCE_REGIME"] = batch.meta["EVIDENCE_REGIME"].to_numpy()
        policy_scope, policy_error_upper = self.policy_.rule_metadata(  # type: ignore[union-attr]
            calibrated,
            out["EVIDENCE_REGIME"].to_numpy(dtype=str),
        )
        out["ACCEPTANCE_POLICY_SCOPE"] = policy_scope
        out["ACCEPTANCE_POLICY_ERROR_UPPER"] = policy_error_upper
        out["POLICY_ACCEPTED"] = policy_accepted
        out["CLASS_CERTIFIED_FOR_HARD_LABEL"] = class_certified
        out["HARD_LABEL_CERTIFIED"] = class_certified
        out["ENCOUNTER_LABEL_CONFLICT"] = encounter_conflict
        out["SRKW_DOMAIN_SUPPORTED"] = srkw_domain_supported
        out["TRANSIENT_DOMAIN_SUPPORTED"] = transient_domain_supported
        out["IMPUTATION_DOMAIN_SUPPORTED"] = predicted_domain_supported
        out["BINARY_MODEL_DOMAIN_SUPPORTED"] = binary_domain_supported
        out["MODEL_DOMAIN_STATUS"] = (
            "CONFIGURED"
            if {"SRKW", "TRANSIENT"}
            <= set(getattr(self, "model_domain_geometries_", {}) or {})
            else "NOT_CONFIGURED"
        )
        out["SOFT_COUNT_CERTIFIED"] = (
            certification_release_ready(soft_mass_certification)
            & certified_stratum_supported
        )
        out["SOFT_CERTIFICATION_STRATUM_SUPPORTED"] = certified_stratum_supported
        for name, values in stability.items():
            out[name] = values
        out["IMPUTATION_APPLIED"] = accepted
        out["ECOTYPE_DETAIL_IMPUTED"] = imputed
        out["ECOTYPE_DETAIL_EFFECTIVE"] = np.where(
            accepted, imputed, out["ECOTYPE_DETAIL"]
        )
        out["ACCEPTANCE_THRESHOLD"] = threshold
        out["ABSTENTION_REASON"] = np.where(accepted, pd.NA, reason)
        out["IMPUTATION_METHOD"] = MODEL_VERSION
        out["MODEL_REGIME"] = self.config.regime
        out["ECOTYPE_LABEL_TIER"] = np.select(
            [
                accepted,
                soft_eligible,
                encounter_conflict,
                unstable | stability_not_evaluated,
                policy_accepted & ~predicted_domain_supported,
                uncertified,
                locally_supported,
            ],
            [
                "IMPUTED_HIGH_CONFIDENCE",
                "PROBABILISTIC_CERTIFIED",
                "UNKNOWN_CONFLICTING_ENCOUNTER",
                "PROBABILISTIC_SENSITIVE",
                "UNKNOWN_OUTSIDE_MODEL_DOMAIN",
                "PROBABILISTIC_UNCERTIFIED",
                "PROBABILISTIC_LOCAL",
            ],
            default="UNKNOWN_UNSUPPORTED",
        )
        probability_eligible = accepted | soft_eligible
        out["EXPECTED_SRKW_COUNT"] = np.where(probability_eligible, calibrated, 0.0)
        out["EXPECTED_TRANSIENT_COUNT"] = np.where(
            probability_eligible, 1.0 - calibrated, 0.0
        )
        out["EXPECTED_OTHER_COUNT"] = 0.0
        out["EXPECTED_UNKNOWN_COUNT"] = (~probability_eligible).astype(float)
        out["USE_FOR_HARD_COUNTS"] = accepted
        out["USE_FOR_PROBABILISTIC_COUNTS"] = probability_eligible
        for column in DIAGNOSTIC_META_COLUMNS:
            out[column] = batch.meta[column].to_numpy()
        validate_imputation_mass(out, query_labels=self.config.query_labels)
        return out

    def apply_to_all(self, observations: pd.DataFrame) -> pd.DataFrame:
        if self.encounter_lookup_.empty:
            raise ValueError("Fitted imputer has no encounter lookup")
        encounter_lookup = self.encounter_lookup_.set_index("OBSERVATION_ID")
        identifiers = observations["OBSERVATION_ID"].astype(str)
        missing_encounters = sorted(
            set(identifiers) - set(encounter_lookup.index.astype(str))
        )
        if missing_encounters:
            raise ValueError(
                "Fitted imputer encounter lookup does not cover the supplied observations: "
                f"{missing_encounters[:10]}"
            )
        predicted = self.predict_queries(observations)
        prediction_lookup = (
            predicted.set_index("OBSERVATION_ID") if not predicted.empty else None
        )
        out = observations.copy()
        out["ENCOUNTER_ID"] = identifiers.map(encounter_lookup["ENCOUNTER_ID"])
        out["ENCOUNTER_SIZE"] = identifiers.map(encounter_lookup["ENCOUNTER_SIZE"])
        out["ENCOUNTER_LABEL_CONFLICT"] = (
            identifiers.map(encounter_lookup["ENCOUNTER_LABEL_CONFLICT"])
            if "ENCOUNTER_LABEL_CONFLICT" in encounter_lookup
            else False
        )
        out["ENCOUNTER_LABEL_CONFLICT"] = (
            out["ENCOUNTER_LABEL_CONFLICT"].fillna(False).astype(bool)
        )
        if out[["ENCOUNTER_ID", "ENCOUNTER_SIZE"]].isna().any().any():
            raise ValueError(
                "Encounter lookup produced missing final encounter assignments"
            )
        detail = out["ECOTYPE_DETAIL"].astype(str).str.upper()
        out["ECOTYPE_DETAIL_OBSERVED"] = detail
        out["ECOTYPE_DETAIL_EFFECTIVE"] = detail
        if "ECOTYPE_BUCKET" in out:
            out["ECOTYPE_BUCKET_EFFECTIVE"] = (
                out["ECOTYPE_BUCKET"].astype(str).str.upper()
            )
        else:
            out["ECOTYPE_BUCKET_EFFECTIVE"] = np.where(
                detail.isin(["SRKW", "TRANSIENT"]), detail, "OTHER"
            )
        out["DISPLAY_CLASS"] = "KNOWN_OTHER"
        out.loc[detail.eq("SRKW"), "DISPLAY_CLASS"] = "SRKW"
        out.loc[detail.eq("TRANSIENT"), "DISPLAY_CLASS"] = "TRANSIENT"
        out.loc[detail.isin(self.config.query_labels), "DISPLAY_CLASS"] = (
            "UNKNOWN_STILL"
        )
        out["P_SRKW"] = np.nan
        out["P_TRANSIENT"] = np.nan
        out["P_OTHER"] = np.nan
        out.loc[detail.eq("SRKW"), ["P_SRKW", "P_TRANSIENT"]] = (1.0, 0.0)
        out.loc[detail.eq("TRANSIENT"), ["P_SRKW", "P_TRANSIENT"]] = (0.0, 1.0)
        binary_observed = detail.isin(self.config.training_labels)
        query_observed = detail.isin(self.config.query_labels)
        known_other = ~(binary_observed | query_observed)
        out.loc[binary_observed, "P_OTHER"] = 0.0
        out.loc[known_other, "P_OTHER"] = 1.0
        out["IMPUTATION_APPLIED"] = False
        out["ECOTYPE_DETAIL_IMPUTED"] = pd.NA
        out["ABSTENTION_REASON"] = pd.NA
        out["EVIDENCE_REGIME"] = pd.NA
        out["IMPUTATION_METHOD"] = pd.NA
        out["IMPUTATION_CONFIDENCE"] = np.nan
        out["IMPUTATION_TIME_DIRECTION"] = "NOT_APPLICABLE"
        out["STABILITY_EVALUATED"] = False
        out["SOFT_COUNT_CERTIFIED"] = False
        out["HARD_LABEL_CERTIFIED"] = False
        out["SRKW_DOMAIN_SUPPORTED"] = False
        out["TRANSIENT_DOMAIN_SUPPORTED"] = False
        out["IMPUTATION_DOMAIN_SUPPORTED"] = False
        out["BINARY_MODEL_DOMAIN_SUPPORTED"] = False
        out["MODEL_DOMAIN_STATUS"] = (
            "CONFIGURED"
            if {"SRKW", "TRANSIENT"}
            <= set(getattr(self, "model_domain_geometries_", {}) or {})
            else "NOT_CONFIGURED"
        )
        observed_training_label = detail.isin(self.config.training_labels)
        out["ELIGIBLE_FOR_TRAINING"] = (
            observed_training_label & ~out["ENCOUNTER_LABEL_CONFLICT"]
        )
        out["ELIGIBLE_FOR_EVALUATION"] = (
            observed_training_label & ~out["ENCOUNTER_LABEL_CONFLICT"]
        )
        out["ECOTYPE_LABEL_TIER"] = np.select(
            [
                detail.isin(self.config.training_labels),
                detail.isin(self.config.query_labels),
            ],
            ["OBSERVED", "UNKNOWN_UNSUPPORTED"],
            default="KNOWN_OTHER",
        )
        out["EXPECTED_SRKW_COUNT"] = detail.eq("SRKW").astype(float)
        out["EXPECTED_TRANSIENT_COUNT"] = detail.eq("TRANSIENT").astype(float)
        out["EXPECTED_OTHER_COUNT"] = known_other.astype(float)
        out["EXPECTED_UNKNOWN_COUNT"] = detail.isin(self.config.query_labels).astype(
            float
        )
        observed_count_eligible = (
            detail.isin(self.config.training_labels) & ~out["ENCOUNTER_LABEL_CONFLICT"]
        )
        out["USE_FOR_HARD_COUNTS"] = observed_count_eligible
        out["USE_FOR_PROBABILISTIC_COUNTS"] = observed_count_eligible
        if prediction_lookup is not None:
            ids = out["OBSERVATION_ID"].astype(str)
            for column in [
                "P_SRKW_RAW",
                "P_SRKW",
                "P_TRANSIENT",
                "P_OTHER",
                "PREDICTED_CLASS",
                "IMPUTATION_APPLIED",
                "ECOTYPE_DETAIL_IMPUTED",
                "ABSTENTION_REASON",
                "EVIDENCE_REGIME",
                "CONFORMAL_SET",
                "CONFORMAL_SET_SIZE",
                "CONFORMAL_P_TRANSIENT",
                "CONFORMAL_P_SRKW",
                "OOD_SCORE",
                "OOD_THRESHOLD",
                "OOD_MARGIN",
                "OOD_INLIER",
                "OTHER_SUPPORT_VETO",
                "ACCEPTANCE_THRESHOLD",
                "ACCEPTANCE_POLICY_SCOPE",
                "ACCEPTANCE_POLICY_ERROR_UPPER",
                "POLICY_ACCEPTED",
                "CLASS_CERTIFIED_FOR_HARD_LABEL",
                "HARD_LABEL_CERTIFIED",
                "ENCOUNTER_LABEL_CONFLICT",
                "SRKW_DOMAIN_SUPPORTED",
                "TRANSIENT_DOMAIN_SUPPORTED",
                "IMPUTATION_DOMAIN_SUPPORTED",
                "BINARY_MODEL_DOMAIN_SUPPORTED",
                "MODEL_DOMAIN_STATUS",
                "SOFT_COUNT_CERTIFIED",
                "SOFT_CERTIFICATION_STRATUM_SUPPORTED",
                "P_SRKW_NO_SAME_DAY",
                "P_SRKW_NO_FUTURE",
                "P_SRKW_NO_NEAREST_ENCOUNTER",
                "MAX_STABILITY_PROBABILITY_SHIFT",
                "STABILITY_CLASS_AGREEMENT",
                "PREDICTION_STABLE",
                "STABILITY_EVALUATED",
                "ECOTYPE_LABEL_TIER",
                "EXPECTED_SRKW_COUNT",
                "EXPECTED_TRANSIENT_COUNT",
                "EXPECTED_OTHER_COUNT",
                "EXPECTED_UNKNOWN_COUNT",
                "USE_FOR_HARD_COUNTS",
                "USE_FOR_PROBABILISTIC_COUNTS",
                *DIAGNOSTIC_META_COLUMNS,
            ]:
                if column in prediction_lookup:
                    mapped = ids.map(prediction_lookup[column])
                    if column in out.columns:
                        out[column] = mapped.combine_first(out[column])
                    else:
                        out[column] = mapped
            applied = (
                out["IMPUTATION_APPLIED"].astype("boolean").fillna(False).astype(bool)
            )
            accepted_srkw = applied & out["ECOTYPE_DETAIL_IMPUTED"].eq("SRKW")
            accepted_transient = applied & out["ECOTYPE_DETAIL_IMPUTED"].eq("TRANSIENT")
            accepted = accepted_srkw | accepted_transient
            out.loc[accepted_srkw, "DISPLAY_CLASS"] = "SRKW_ASSIGNED"
            out.loc[accepted_transient, "DISPLAY_CLASS"] = "TRANSIENT_ASSIGNED"
            out.loc[accepted, "ECOTYPE_DETAIL_EFFECTIVE"] = out.loc[
                accepted, "ECOTYPE_DETAIL_IMPUTED"
            ]
            out.loc[accepted, "ECOTYPE_BUCKET_EFFECTIVE"] = out.loc[
                accepted, "ECOTYPE_DETAIL_IMPUTED"
            ]
            out.loc[accepted, "IMPUTATION_METHOD"] = MODEL_VERSION
            out.loc[accepted, "IMPUTATION_CONFIDENCE"] = out.loc[
                accepted, ["P_SRKW", "P_TRANSIENT"]
            ].max(axis=1)
            out.loc[accepted, "IMPUTATION_TIME_DIRECTION"] = (
                "PAST_AND_FUTURE"
                if self.config.regime == "retrospective"
                else "PAST_AND_SAME_DAY"
            )
            # Pseudo-labels are outputs, never independent observed training truth.
            out.loc[accepted, "ELIGIBLE_FOR_TRAINING"] = False
            out.loc[accepted, "ELIGIBLE_FOR_EVALUATION"] = False

        for column in (
            "ENCOUNTER_LABEL_CONFLICT",
            "SRKW_DOMAIN_SUPPORTED",
            "TRANSIENT_DOMAIN_SUPPORTED",
            "IMPUTATION_DOMAIN_SUPPORTED",
            "BINARY_MODEL_DOMAIN_SUPPORTED",
            "SOFT_COUNT_CERTIFIED",
            "HARD_LABEL_CERTIFIED",
            "STABILITY_EVALUATED",
        ):
            out[column] = out[column].fillna(False).astype(bool)
        sighting_dates = pd.to_datetime(
            out["SIGHTING_DATE"], errors="coerce"
        ).dt.normalize()
        latest_date = sighting_dates.max()
        is_query = detail.isin(self.config.query_labels)
        if self.config.regime == "retrospective" and pd.notna(latest_date):
            complete_through = latest_date - pd.Timedelta(
                days=int(self.config.feature.max_day_lag)
            )
            context_complete = sighting_dates.le(complete_through)
            out["IMPUTATION_CONTEXT_STATUS"] = np.select(
                [~is_query, context_complete],
                ["OBSERVED", "MATURE_RETROSPECTIVE"],
                default="PROVISIONAL_RECENT",
            )
        else:
            complete_through = latest_date
            context_complete = pd.Series(True, index=out.index)
            out["IMPUTATION_CONTEXT_STATUS"] = np.where(
                is_query, "OPERATIONAL_EOD", "OBSERVED"
            )
        out["RETROSPECTIVE_CONTEXT_COMPLETE"] = context_complete.fillna(False).astype(
            bool
        )
        out["CONTEXT_COMPLETE_THROUGH_DATE"] = (
            complete_through.date() if pd.notna(complete_through) else pd.NaT
        )
        validate_imputation_mass(out, query_labels=self.config.query_labels)
        return out

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        import os, tempfile

        fd, temporary = tempfile.mkstemp(prefix=p.name + ".", dir=p.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"MMTK-IMPUTER\x00\x01\n")
                joblib.dump(self, handle)
            os.replace(temporary, p)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "SelectiveDateContextImputer":
        with Path(path).open("rb") as handle:
            if handle.read(len(b"MMTK-IMPUTER\x00\x01\n")) != b"MMTK-IMPUTER\x00\x01\n":
                raise ValueError(
                    "Legacy or unsupported imputation model. Refit with marine-mammals killer-whales observations impute fit; existing model files are unchanged."
                )
            loaded = joblib.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(loaded).__name__}")
        # Binary model artifacts never embed authority to release soft counts.
        # A future open-set implementation must validate an immutable sidecar
        # against exact scorer bytes at application time.
        loaded.soft_mass_certification_ = {"CERTIFIED": False, "EVIDENCE_ID": None}
        if not hasattr(loaded, "model_domain_geometries_"):
            loaded.model_domain_geometries_ = {}
        if not hasattr(loaded, "model_domain_provenance_"):
            loaded.model_domain_provenance_ = {}
        if not hasattr(loaded, "encounter_quarantine_summary_"):
            loaded.encounter_quarantine_summary_ = {}
        return loaded
