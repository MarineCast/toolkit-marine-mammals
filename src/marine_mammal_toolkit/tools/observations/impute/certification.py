from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

SOFT_COUNT_CERTIFICATION_SCHEMA_VERSION = 1
_BINDING_FIELDS = (
    "model_version",
    "fit_run_id",
    "training_data_snapshot_id",
    "imputation_config_sha256",
    "source_config_sha256",
    "model_domain_sha256",
    "model_identity_sha256",
    "open_set_scorer_sha256",
)


def _load_evidence(evidence: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(evidence, Mapping):
        return dict(evidence)
    path = Path(evidence)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read soft-count certification evidence: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("Soft-count certification evidence must be a JSON object")
    return value


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Soft-count evidence requires object {key!r}")
    return value


def _required_true(parent: Mapping[str, Any], key: str, *, scope: str) -> None:
    if parent.get(key) is not True:
        raise ValueError(f"Soft-count evidence requires {scope}.{key}=true")


def _finite_number(parent: Mapping[str, Any], key: str, *, scope: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Soft-count evidence requires numeric {scope}.{key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Soft-count evidence requires numeric {scope}.{key}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Soft-count evidence requires finite {scope}.{key}")
    return number


def _nonnegative_integer(parent: Mapping[str, Any], key: str, *, scope: str) -> int:
    number = _finite_number(parent, key, scope=scope)
    if number < 0 or not number.is_integer():
        raise ValueError(
            f"Soft-count evidence requires nonnegative integer {scope}.{key}"
        )
    return int(number)


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def certification_release_ready(certification: Mapping[str, Any] | None) -> bool:
    """Return true only for a fully bound, open-set-capable certification."""

    value = certification or {}
    binding = value.get("BINDING")
    strata = value.get("ELIGIBLE_STRATA")
    valid_binding = isinstance(binding, Mapping) and all(
        isinstance(binding.get(key), str) and str(binding.get(key)).strip()
        for key in _BINDING_FIELDS
    )
    if valid_binding:
        valid_binding = all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(binding[key])) is not None
            for key in _BINDING_FIELDS
            if key.endswith("_sha256")
        )
    valid_strata = (
        isinstance(strata, list)
        and bool(strata)
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("source"), str)
            and bool(str(item.get("source")).strip())
            and isinstance(item.get("era_start"), str)
            and isinstance(item.get("era_end"), str)
            for item in (strata or [])
        )
    )
    return bool(
        value.get("CERTIFIED") is True
        and value.get("EVIDENCE_GATES_PASSED") is True
        and value.get("SCHEMA_VERSION") == SOFT_COUNT_CERTIFICATION_SCHEMA_VERSION
        and value.get("BINDING_VALIDATED") is True
        and value.get("OPEN_SET_SCORER_BOUND") is True
        and value.get("OPEN_SET_OUTPUT_ENABLED") is True
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("EVIDENCE_SHA256")))
        is not None
        and valid_binding
        and valid_strata
    )


def validate_soft_count_certification(
    evidence: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate independent evidence before releasing fractional counts.

    The evaluation producer must emit this schema from frozen,
    encounter-independent outer folds. Missing evidence is a release failure,
    never an implicit pass.
    """

    document = _load_evidence(evidence)
    if document.get("schema_version") != SOFT_COUNT_CERTIFICATION_SCHEMA_VERSION:
        raise ValueError(
            "Soft-count certification schema_version must be "
            f"{SOFT_COUNT_CERTIFICATION_SCHEMA_VERSION}"
        )
    evidence_id = document.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("Soft-count certification evidence_id must be non-empty")

    binding = _mapping(document, "binding")
    normalized_binding: dict[str, str] = {}
    for key in _BINDING_FIELDS:
        value = binding.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Soft-count evidence requires non-empty binding.{key}")
        normalized_binding[key] = value.strip()
    for key in _BINDING_FIELDS:
        if (
            key.endswith("_sha256")
            and re.fullmatch(r"sha256:[0-9a-f]{64}", normalized_binding[key]) is None
        ):
            raise ValueError(
                f"Soft-count evidence binding.{key} must use sha256:<digest>"
            )

    design = _mapping(document, "evaluation_design")
    for key in (
        "natural_deployment_prevalence",
        "nested_model_selection",
        "separate_classifier_calibration_policy_sets",
        "required_strata_complete",
    ):
        _required_true(design, key, scope="evaluation_design")
    if design.get("outer_test_balanced_or_capped") is not False:
        raise ValueError(
            "Soft-count evidence requires evaluation_design.outer_test_balanced_or_capped=false"
        )
    if (
        _nonnegative_integer(design, "rolling_origin_folds", scope="evaluation_design")
        < 5
    ):
        raise ValueError(
            "Soft-count evidence requires at least five rolling-origin folds"
        )
    if (
        _nonnegative_integer(design, "lineage_leakage_count", scope="evaluation_design")
        != 0
    ):
        raise ValueError(
            "Soft-count evidence must report zero target/duplicate lineage leakage"
        )

    bootstrap = _mapping(document, "paired_encounter_bootstrap")
    for comparison in (
        "brier_vs_prevalence",
        "brier_vs_local_support",
        "log_loss_vs_prevalence",
        "log_loss_vs_local_support",
    ):
        result = _mapping(bootstrap, comparison)
        if (
            _finite_number(
                result,
                "improvement_ci95_lower",
                scope=f"paired_encounter_bootstrap.{comparison}",
            )
            <= 0
        ):
            raise ValueError(
                f"Soft-count evidence requires positive paired improvement for {comparison}"
            )

    calibration = _mapping(document, "calibration")
    overall = _mapping(calibration, "overall")
    if abs(_finite_number(overall, "intercept", scope="calibration.overall")) > 0.10:
        raise ValueError("Overall calibration intercept exceeds +/-0.10")
    slope = _finite_number(overall, "slope", scope="calibration.overall")
    if not 0.8 <= slope <= 1.2:
        raise ValueError("Overall calibration slope must be within [0.8, 1.2]")
    if _finite_number(overall, "equal_mass_ece", scope="calibration.overall") > 0.03:
        raise ValueError("Overall equal-mass ECE exceeds 0.03")
    if (
        _nonnegative_integer(overall, "independent_units", scope="calibration.overall")
        < 100
    ):
        raise ValueError("Overall calibration requires at least 100 independent units")

    strata = calibration.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("Soft-count evidence requires source/era calibration strata")
    eligible_strata: list[dict[str, Any]] = []
    abstained_strata: list[dict[str, Any]] = []
    for index, raw_stratum in enumerate(strata):
        if not isinstance(raw_stratum, Mapping):
            raise ValueError(f"Calibration stratum {index} must be an object")
        scope = f"calibration.strata[{index}]"
        source = raw_stratum.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Calibration stratum {index} requires source")
        try:
            era_start = date.fromisoformat(str(raw_stratum.get("era_start")))
            era_end = date.fromisoformat(str(raw_stratum.get("era_end")))
        except ValueError as exc:
            raise ValueError(
                f"Calibration stratum {index} requires ISO era bounds"
            ) from exc
        if era_start > era_end:
            raise ValueError(f"Calibration stratum {index} has reversed era bounds")
        units = _nonnegative_integer(raw_stratum, "independent_units", scope=scope)
        abstained = raw_stratum.get("abstained")
        selector = {
            "source": source.strip().upper(),
            "era_start": era_start.isoformat(),
            "era_end": era_end.isoformat(),
            "independent_units": units,
        }
        if units < 100:
            if abstained is not True:
                raise ValueError(
                    f"Insufficient calibration stratum {index} must abstain"
                )
            abstained_strata.append(selector)
            continue
        if abstained is not False:
            raise ValueError(
                f"Sufficient calibration stratum {index} cannot be abstained"
            )
        if _finite_number(raw_stratum, "equal_mass_ece", scope=scope) > 0.05:
            raise ValueError(f"Calibration stratum {index} equal-mass ECE exceeds 0.05")
        eligible_strata.append(selector)
    if not eligible_strata:
        raise ValueError(
            "Soft-count evidence requires at least one non-abstained stratum"
        )

    audit = _mapping(document, "unknown_label_audit")
    for key in ("blinded", "double_reviewed", "expected_totals_within_95_interval"):
        _required_true(audit, key, scope="unknown_label_audit")
    if (
        not isinstance(audit.get("sample_id"), str)
        or not str(audit["sample_id"]).strip()
    ):
        raise ValueError("Soft-count evidence requires unknown_label_audit.sample_id")
    if (
        _nonnegative_integer(audit, "independent_units", scope="unknown_label_audit")
        < 1
    ):
        raise ValueError("Unknown-label audit must contain independent reviewed units")

    open_set = _mapping(document, "open_set")
    for key in (
        "unconditional_probabilities",
        "unit_mass_conserved",
        "domain_gate_enforced",
    ):
        _required_true(open_set, key, scope="open_set")
    if _finite_number(open_set, "other_recall", scope="open_set") < 0.90:
        raise ValueError("Open-set Other recall is below 0.90")
    if _nonnegative_integer(open_set, "independent_other_units", scope="open_set") < 1:
        raise ValueError("Open-set evaluation requires independent Other units")

    stability = _mapping(document, "stability")
    eligible = _nonnegative_integer(
        stability, "probability_eligible_rows", scope="stability"
    )
    evaluated = _nonnegative_integer(stability, "evaluated_rows", scope="stability")
    if eligible < 1 or evaluated != eligible:
        raise ValueError("Every probability-eligible row must have stability evaluated")
    if _finite_number(stability, "maximum_probability_shift", scope="stability") > 0.20:
        raise ValueError("Maximum stability probability shift exceeds 0.20")
    if _nonnegative_integer(stability, "class_flip_count", scope="stability") != 0:
        raise ValueError("Stability evaluation contains class flips")

    return {
        # Evidence thresholds passing is necessary but not sufficient. The
        # imputer must still bind these claims to its exact fitted identity and
        # prove that unconditional open-set output is implemented.
        "CERTIFIED": False,
        "EVIDENCE_GATES_PASSED": True,
        "EVIDENCE_ID": evidence_id.strip(),
        "EVIDENCE_SHA256": _canonical_sha256(document),
        "SCHEMA_VERSION": SOFT_COUNT_CERTIFICATION_SCHEMA_VERSION,
        "BINDING": normalized_binding,
        "ELIGIBLE_STRATA": eligible_strata,
        "ABSTAINED_STRATA": abstained_strata,
        "BINDING_VALIDATED": False,
        "OPEN_SET_SCORER_BOUND": False,
        "OPEN_SET_OUTPUT_ENABLED": False,
    }
