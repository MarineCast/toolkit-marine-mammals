from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from marine_mammal_toolkit.tools.schemas.observations import ASSOCIATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import OBSERVATION_SCHEMA_V8
from marine_mammal_toolkit.tools.observations.impute.certification import (
    validate_soft_count_certification,
)
from marine_mammal_toolkit.tools.observations.impute.encounters import (
    attach_encounter_ids,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputer import (
    KillerWhaleImputer as SelectiveDateContextImputer,
)
from marine_mammal_toolkit.tools.observations.impute.model import (
    certified_soft_mass_mask,
)
from marine_mammal_toolkit.tools.observations.impute.model import (
    mark_encounter_label_conflicts,
)
from marine_mammal_toolkit.tools.observations.impute.model import (
    validate_imputation_mass,
)
from marine_mammal_toolkit.tools.observations.impute.pipeline import (
    _normalization_snapshot_id,
)
from marine_mammal_toolkit.tools.observations.impute.pipeline import (
    _stratified_oof_summary,
)
from marine_mammal_toolkit.tools.observations.impute.pipeline import resolve_model_path
from marine_mammal_toolkit.tools.observations.impute.splits import (
    make_spatiotemporal_groups,
)


def _encounter_frame(longitudes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OBSERVATION_ID": [
                chr(ord("a") + index) for index in range(len(longitudes))
            ],
            "SIGHTING_DATE": ["2025-06-03"] * len(longitudes),
            "SOURCE_EVENT_AT_UTC": [pd.NaT] * len(longitudes),
            "SOURCE_TIME_PRECISION": ["DATE"] * len(longitudes),
            "LATITUDE": [48.5] * len(longitudes),
            "LONGITUDE": longitudes,
        }
    )


def _valid_soft_count_evidence() -> dict:
    improvement = {"improvement_ci95_lower": 0.01}
    digest = "sha256:" + ("a" * 64)
    return {
        "schema_version": 1,
        "evidence_id": "soft-eval-2026-08",
        "binding": {
            "model_version": "open-set-v1",
            "fit_run_id": "fit-v1",
            "training_data_snapshot_id": "snapshot-v1",
            "imputation_config_sha256": digest,
            "source_config_sha256": digest,
            "model_domain_sha256": digest,
            "model_identity_sha256": digest,
            "open_set_scorer_sha256": digest,
        },
        "evaluation_design": {
            "natural_deployment_prevalence": True,
            "nested_model_selection": True,
            "separate_classifier_calibration_policy_sets": True,
            "required_strata_complete": True,
            "outer_test_balanced_or_capped": False,
            "rolling_origin_folds": 5,
            "lineage_leakage_count": 0,
        },
        "paired_encounter_bootstrap": {
            "brier_vs_prevalence": improvement,
            "brier_vs_local_support": improvement,
            "log_loss_vs_prevalence": improvement,
            "log_loss_vs_local_support": improvement,
        },
        "calibration": {
            "overall": {
                "intercept": 0.01,
                "slope": 1.0,
                "equal_mass_ece": 0.02,
                "independent_units": 500,
            },
            "strata": [
                {
                    "source": "TWM",
                    "era_start": "2020-01-01",
                    "era_end": "2025-12-31",
                    "independent_units": 150,
                    "equal_mass_ece": 0.04,
                    "abstained": False,
                },
                {
                    "source": "CWR",
                    "era_start": "2020-01-01",
                    "era_end": "2025-12-31",
                    "independent_units": 80,
                    "abstained": True,
                },
            ],
        },
        "unknown_label_audit": {
            "sample_id": "unknown-audit-v1",
            "independent_units": 100,
            "blinded": True,
            "double_reviewed": True,
            "expected_totals_within_95_interval": True,
        },
        "open_set": {
            "other_recall": 0.91,
            "independent_other_units": 100,
            "unconditional_probabilities": True,
            "unit_mass_conserved": True,
            "domain_gate_enforced": True,
        },
        "stability": {
            "probability_eligible_rows": 200,
            "evaluated_rows": 200,
            "maximum_probability_shift": 0.19,
            "class_flip_count": 0,
        },
    }


def test_complete_link_encounters_prevent_date_only_spatial_chaining():
    frame = _encounter_frame([-123.20, -123.17, -123.14])
    first = attach_encounter_ids(frame).set_index("OBSERVATION_ID")
    reordered = attach_encounter_ids(frame.iloc[::-1]).set_index("OBSERVATION_ID")

    assert first.at["a", "ENCOUNTER_ID"] == first.at["b", "ENCOUNTER_ID"]
    assert first.at["a", "ENCOUNTER_ID"] != first.at["c", "ENCOUNTER_ID"]
    assert first["ENCOUNTER_SIZE"].max() == 2
    pd.testing.assert_series_equal(
        first["ENCOUNTER_ID"].sort_index(), reordered["ENCOUNTER_ID"].sort_index()
    )


def test_only_trusted_nonconflicting_associations_expand_encounter_radius():
    frame = _encounter_frame([-123.20, -123.08])
    associations = pd.DataFrame(
        {
            "OBSERVATION_ID": ["a", "b"],
            "ASSOCIATION_KIND": ["POD", "POD"],
            "ASSOCIATION_VALUE": ["J", "J"],
            "CONFIDENCE": ["STRONG", "STRONG"],
            "CONFLICTING": [False, False],
        }
    )
    trusted = attach_encounter_ids(frame, associations).set_index("OBSERVATION_ID")
    assert trusted.at["a", "ENCOUNTER_ID"] == trusted.at["b", "ENCOUNTER_ID"]

    associations["CONFLICTING"] = True
    conflicting = attach_encounter_ids(frame, associations).set_index("OBSERVATION_ID")
    assert conflicting.at["a", "ENCOUNTER_ID"] != conflicting.at["b", "ENCOUNTER_ID"]


def test_precise_time_tolerance_still_caps_trusted_associations():
    frame = _encounter_frame([-123.20, -123.08])
    frame["SOURCE_TIME_PRECISION"] = "TIMESTAMP"
    frame["SOURCE_EVENT_AT_UTC"] = pd.to_datetime(
        ["2025-06-03T01:00:00Z", "2025-06-03T12:00:00Z"]
    )
    associations = pd.DataFrame(
        {
            "OBSERVATION_ID": ["a", "b"],
            "ASSOCIATION_KIND": ["POD", "POD"],
            "ASSOCIATION_VALUE": ["J", "J"],
            "CONFIDENCE": ["STRONG", "STRONG"],
            "CONFLICTING": [False, False],
        }
    )
    result = attach_encounter_ids(frame, associations).set_index("OBSERVATION_ID")
    assert result.at["a", "ENCOUNTER_ID"] != result.at["b", "ENCOUNTER_ID"]


def test_mixed_ecotype_conflict_quarantines_entire_encounter():
    frame = pd.DataFrame(
        {
            "ENCOUNTER_ID": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "ECOTYPE_DETAIL": [
                "SRKW",
                "TRANSIENT",
                "MIXED",
                "UNKNOWN",
                "SRKW",
                "UNKNOWN",
            ],
            "LABEL_CONFLICT": [False] * 6,
        }
    )
    result = mark_encounter_label_conflicts(frame)
    assert result.groupby("ENCOUNTER_ID")[
        "ENCOUNTER_LABEL_CONFLICT"
    ].first().to_dict() == {
        "e1": True,
        "e2": True,
        "e3": False,
    }


def test_mixed_ecotype_conflict_is_ineligible_for_every_model_and_count_path():
    observations = pd.DataFrame(
        {
            "OBSERVATION_ID": ["srkw", "transient"],
            "SIGHTING_DATE": ["2025-06-03", "2025-06-03"],
            "ECOTYPE_DETAIL": ["SRKW", "TRANSIENT"],
            "ECOTYPE_BUCKET": ["SRKW", "TRANSIENT"],
        }
    )
    imputer = object.__new__(SelectiveDateContextImputer)
    imputer.encounter_lookup_ = pd.DataFrame(
        {
            "OBSERVATION_ID": observations["OBSERVATION_ID"],
            "ENCOUNTER_ID": ["conflict", "conflict"],
            "ENCOUNTER_SIZE": [2, 2],
            "ENCOUNTER_LABEL_CONFLICT": [True, True],
        }
    )
    imputer.config = SimpleNamespace(
        training_labels=("SRKW", "TRANSIENT"),
        query_labels=("UNKNOWN",),
        regime="retrospective",
        feature=SimpleNamespace(max_day_lag=14),
    )
    imputer.model_domain_geometries_ = {}
    imputer.predict_queries = lambda _: pd.DataFrame()

    result = imputer.apply_to_all(observations)

    assert not result[
        [
            "ELIGIBLE_FOR_TRAINING",
            "ELIGIBLE_FOR_EVALUATION",
            "USE_FOR_HARD_COUNTS",
            "USE_FOR_PROBABILISTIC_COUNTS",
        ]
    ].any(axis=None)


def test_spatiotemporal_groups_never_split_an_encounter():
    frame = pd.DataFrame(
        {
            "ENCOUNTER_ID": ["same", "same", "other"],
            "SIGHTING_DATE": ["2025-01-01", "2025-01-02", "2025-02-01"],
            "LATITUDE": [48.0, 49.0, 47.0],
            "LONGITUDE": [-123.0, -125.0, -122.0],
        }
    )
    groups = make_spatiotemporal_groups(frame)
    assert groups.iloc[0] == groups.iloc[1]


def test_soft_count_mass_requires_every_gate():
    yes = np.asarray([True])
    no = np.asarray([False])
    common = {
        "policy_accepted": yes,
        "locally_supported": yes,
        "binary_domain_supported": yes,
        "stability_evaluated": yes,
        "prediction_stable": yes,
        "encounter_conflict": no,
        "certified_stratum_supported": yes,
        "open_set_probability_mass_validated": yes,
    }
    assert not certified_soft_mass_mask(**common, certification=None).item()
    ready = validate_soft_count_certification(_valid_soft_count_evidence())
    ready.update(
        {
            "CERTIFIED": True,
            "BINDING_VALIDATED": True,
            "OPEN_SET_SCORER_BOUND": True,
            "OPEN_SET_OUTPUT_ENABLED": True,
        }
    )
    assert certified_soft_mass_mask(**common, certification=ready).item()
    assert not certified_soft_mass_mask(
        **{**common, "stability_evaluated": no},
        certification=ready,
    ).item()
    assert not certified_soft_mass_mask(
        **{**common, "certified_stratum_supported": no},
        certification=ready,
    ).item()
    assert not certified_soft_mass_mask(
        **{**common, "open_set_probability_mass_validated": no},
        certification=ready,
    ).item()
    assert not certified_soft_mass_mask(
        **common,
        certification={"CERTIFIED": True, "EVIDENCE_ID": "legacy-bypass"},
    ).item()


def test_enabling_soft_count_mass_requires_validated_evidence():
    imputer = object.__new__(SelectiveDateContextImputer)
    imputer.training_summary_ = {}
    with pytest.raises(ValueError, match="certification evidence"):
        imputer.set_soft_mass_certification(certified=True)
    evidence = validate_soft_count_certification(_valid_soft_count_evidence())
    assert evidence["EVIDENCE_GATES_PASSED"] is True
    assert evidence["CERTIFIED"] is False
    with pytest.raises(NotImplementedError, match="binary imputer"):
        imputer.set_soft_mass_certification(
            certified=True, evidence=_valid_soft_count_evidence()
        )


def test_soft_count_certification_rejects_failed_stratum():
    evidence = _valid_soft_count_evidence()
    evidence["calibration"]["strata"][0]["equal_mass_ece"] = 0.051
    imputer = object.__new__(SelectiveDateContextImputer)
    imputer.training_summary_ = {}
    with pytest.raises(ValueError, match="stratum 0"):
        imputer.set_soft_mass_certification(certified=True, evidence=evidence)


def test_soft_count_certification_rejects_all_abstained_strata():
    evidence = _valid_soft_count_evidence()
    evidence["calibration"]["strata"] = [
        {
            "source": "CWR",
            "era_start": "2020-01-01",
            "era_end": "2025-12-31",
            "independent_units": 80,
            "abstained": True,
        }
    ]
    with pytest.raises(ValueError, match="non-abstained"):
        validate_soft_count_certification(evidence)


def test_loading_binary_model_discards_forged_soft_certification(tmp_path):
    forged = validate_soft_count_certification(_valid_soft_count_evidence())
    forged.update(
        {
            "CERTIFIED": True,
            "BINDING_VALIDATED": True,
            "OPEN_SET_SCORER_BOUND": True,
            "OPEN_SET_OUTPUT_ENABLED": True,
        }
    )
    imputer = SelectiveDateContextImputer()
    imputer.soft_mass_certification_ = forged
    path = tmp_path / "forged.joblib"
    with path.open("wb") as handle:
        handle.write(b"MMTK-IMPUTER\x00\x01\n")
        joblib.dump(imputer, handle)

    loaded = SelectiveDateContextImputer.load(path)
    assert loaded.soft_mass_certification_ == {
        "CERTIFIED": False,
        "EVIDENCE_ID": None,
    }


def test_oof_diagnostics_report_independent_source_era_strata():
    oof = pd.DataFrame(
        {
            "IS_ENCOUNTER_REPRESENTATIVE": [True, True, False],
            "SOURCE": ["CWR", "CWR", "CWR"],
            "ERA": ["2020-2024"] * 3,
            "SPATIAL_REGION": ["48:-123"] * 3,
            "EVIDENCE_REGIME": ["LAGGED"] * 3,
            "Y_TRUE": [0, 1, 1],
            "P_SRKW": [0.1, 0.8, 0.9],
            "ACCEPTED": [True, True, True],
        }
    )
    summary = _stratified_oof_summary(oof)
    source = summary[summary.STRATIFICATION.eq("SOURCE")].iloc[0]
    assert source["INDEPENDENT_ENCOUNTER_N"] == 2
    assert source["SOURCE"] == "CWR"
    assert source["ACCEPTED_ERROR"] == 0.0


def test_probability_mass_contract_includes_explicit_other_and_unknown():
    frame = pd.DataFrame(
        {
            "ECOTYPE_DETAIL_OBSERVED": ["SRKW", "NRKW", "UNKNOWN", "UNKNOWN"],
            "P_SRKW": [1.0, np.nan, 0.7, 0.4],
            "P_TRANSIENT": [0.0, np.nan, 0.3, 0.6],
            "P_OTHER": [0.0, 1.0, np.nan, np.nan],
            "EXPECTED_SRKW_COUNT": [1.0, 0.0, 0.7, 0.0],
            "EXPECTED_TRANSIENT_COUNT": [0.0, 0.0, 0.3, 0.0],
            "EXPECTED_OTHER_COUNT": [0.0, 1.0, 0.0, 0.0],
            "EXPECTED_UNKNOWN_COUNT": [0.0, 0.0, 0.0, 1.0],
            "USE_FOR_HARD_COUNTS": [True, False, False, False],
            "USE_FOR_PROBABILISTIC_COUNTS": [True, False, True, False],
            "SOFT_COUNT_CERTIFIED": [False, False, True, False],
            "STABILITY_EVALUATED": [False, False, True, False],
        }
    )
    validate_imputation_mass(frame)

    frame.loc[2, "SOFT_COUNT_CERTIFIED"] = False
    with pytest.raises(ValueError, match="explicit certification"):
        validate_imputation_mass(frame)


def test_latest_model_pointer_validates_checksum(tmp_path):
    model = tmp_path / "run" / "ecotype_imputer.joblib"
    model.parent.mkdir()
    model.write_bytes(b"model-v1")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    pointer = tmp_path / "latest.json"
    pointer.write_text(
        json.dumps(
            {"model_path": "run/ecotype_imputer.joblib", "model_sha256": digest}
        ),
        encoding="utf-8",
    )
    assert resolve_model_path(pointer) == model

    model.write_bytes(b"model-v2")
    with pytest.raises(ValueError, match="checksum"):
        resolve_model_path(pointer)


def test_normalization_pointer_validates_checksum_rows_and_schema(tmp_path):
    observations = tmp_path / "observations.parquet"
    associations = tmp_path / "associations.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=OBSERVATION_SCHEMA_V8), observations)
    pq.write_table(pa.Table.from_pylist([], schema=ASSOCIATION_SCHEMA), associations)
    manifest_dir = tmp_path / "manifests" / "normalize"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "run.json"
    snapshot = {"snapshot_id": "snapshot-1"}
    outputs = []
    for dataset_id, path in (
        ("whale.sightings.observations", observations),
        ("whale.sightings.associations", associations),
    ):
        outputs.append(
            {
                "dataset_id": dataset_id,
                "path": str(path),
                "checksum": checksum_path(path),
                "schema_version": "8",
                "file_count": 1,
                "row_count": 0,
                "data_snapshot": snapshot,
            }
        )
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": "8",
                "outputs": outputs,
                "data_snapshot": snapshot,
            }
        ),
        encoding="utf-8",
    )
    (manifest_dir / "latest.json").write_text(
        json.dumps({"manifest": "run.json"}), encoding="utf-8"
    )
    assert _normalization_snapshot_id(observations, associations) == "snapshot-1"

    outputs[0]["row_count"] = 1
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": "8",
                "outputs": outputs,
                "data_snapshot": snapshot,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row count"):
        _normalization_snapshot_id(observations, associations)
