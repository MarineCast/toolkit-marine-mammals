from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from marine_mammal_toolkit.tools.observations.impute.report import (
    build_imputation_report_payload,
)
from marine_mammal_toolkit.tools.observations.impute.report import write_imputation_html


def _imputer() -> SimpleNamespace:
    metrics = {
        "N": 100,
        "BRIER": 0.08,
        "LOG_LOSS": 0.27,
        "ROC_AUC": 0.94,
        "ECE_10": 0.03,
        "COVERAGE": 0.62,
        "SELECTIVE_ACCURACY": 0.97,
        "SELECTIVE_ERROR_UPPER": 0.049,
        "RAW_PROBABILITY_METRICS": {
            "N": 100,
            "BRIER": 0.10,
            "LOG_LOSS": 0.31,
            "ROC_AUC": 0.94,
            "ECE_10": 0.07,
        },
    }
    return SimpleNamespace(
        training_summary_={
            "FIT_RUN_ID": "fit-1",
            "FIT_AT_UTC": "2026-08-19T00:00:00+00:00",
            "MODEL_VERSION": "test-model",
            "TRAINING_DATA_SNAPSHOT_ID": "snapshot-1",
            "LABELED_N": 120,
            "CLASSIFIER_TRAINING_N": 100,
            "LABELED_ENCOUNTER_N": 90,
            "FEATURE_N": 12,
            "FINAL_CALIBRATION_STRATEGY": "encounter",
            "CLASS_CERTIFICATION": {
                "SRKW": {"CERTIFIED": True, "OUTER_ERROR_UPPER": 0.04},
                "TRANSIENT": {"CERTIFIED": True, "OUTER_ERROR_UPPER": 0.03},
            },
        },
        config=SimpleNamespace(
            model=SimpleNamespace(
                final_calibration_strategy="encounter",
                calibration_method="sigmoid",
            )
        ),
        evaluations_={"encounter": SimpleNamespace(metrics=metrics)},
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ECOTYPE_DETAIL_OBSERVED": [
                "SRKW",
                "UNKNOWN",
                "UNKNOWN",
                "UNKNOWN",
                "UNKNOWN",
            ],
            "ECOTYPE_DETAIL_EFFECTIVE": [
                "SRKW",
                "SRKW",
                "TRANSIENT",
                "UNKNOWN",
                "UNKNOWN",
            ],
            "IMPUTATION_APPLIED": [False, True, True, False, False],
            "P_SRKW": [1.0, 0.93, 0.08, 0.55, None],
            "P_TRANSIENT": [0.0, 0.07, 0.92, 0.45, None],
        }
    )


def test_imputation_report_separates_hard_and_probability_scored_points(tmp_path):
    payload = build_imputation_report_payload(
        imputer=_imputer(),
        predictions=_predictions(),
        metadata={"predictions_path": "/tmp/predictions.parquet"},
    )

    assert payload["model_fit"]["fit_run_id"] == "fit-1"
    assert payload["calibration"]["raw"]["ece_10"] == 0.07
    assert payload["calibration"]["calibrated"]["ece_10"] == 0.03
    assert payload["imputation"]["hard_imputed_points"] == 2
    assert payload["imputation"]["hard_imputed_points_by_ecotype"] == {
        "SRKW": 1,
        "TRANSIENT": 1,
    }
    assert payload["imputation"]["probability_scored_unknown_points"] == 3
    assert payload["imputation"]["hard_abstained_unknown_points"] == 2
    json.dumps(payload)

    report = write_imputation_html(tmp_path / "report.html", payload=payload)
    rendered = report.read_text(encoding="utf-8")
    assert "Imputation report" in rendered
    assert "Hard SRKW" in rendered
    assert "Probability-scored points are reported separately" in rendered
