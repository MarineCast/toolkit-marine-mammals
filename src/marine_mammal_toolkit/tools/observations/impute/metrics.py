from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def wilson_upper_bound(errors: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = errors / total
    denominator = 1 + z**2 / total
    center = p + z**2 / (2 * total)
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return float(min(1.0, (center + spread) / denominator))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(p, bins, right=True) - 1, 0, n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        mask = bin_id == index
        if not mask.any():
            continue
        rows.append(
            {
                "BIN": index,
                "LOWER": float(bins[index]),
                "UPPER": float(bins[index + 1]),
                "COUNT": int(mask.sum()),
                "MEAN_P_SRKW": float(p[mask].mean()),
                "OBSERVED_SRKW_RATE": float(y[mask].mean()),
                "ABS_GAP": float(abs(p[mask].mean() - y[mask].mean())),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = reliability_table(y, p, n_bins=n_bins)
    if table.empty:
        return float("nan")
    return float((table["COUNT"] * table["ABS_GAP"]).sum() / table["COUNT"].sum())


def risk_coverage_curve(
    y: np.ndarray,
    p: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    confidence: float = 0.95,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    max_probability = np.maximum(p, 1 - p)
    eligible_mask = (
        np.ones(len(y), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    thresholds = (
        np.linspace(0.50, 0.995, 100) if thresholds is None else np.asarray(thresholds)
    )
    rows = []
    for threshold in thresholds:
        accepted = eligible_mask & (max_probability >= threshold)
        n = int(accepted.sum())
        errors = int((pred[accepted] != y[accepted]).sum()) if n else 0
        rows.append(
            {
                "THRESHOLD": float(threshold),
                "ACCEPTED": n,
                "COVERAGE": float(n / len(y)) if len(y) else 0.0,
                "SELECTIVE_ERROR": float(errors / n) if n else np.nan,
                "SELECTIVE_ACCURACY": float(1 - errors / n) if n else np.nan,
                "ERROR_UPPER": wilson_upper_bound(errors, n, confidence),
            }
        )
    return pd.DataFrame(rows)


def summarize_binary_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    accepted: np.ndarray | None = None,
    confidence: float = 0.95,
) -> dict[str, float | int | dict]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    accepted_mask = (
        np.ones(len(y), dtype=bool)
        if accepted is None
        else np.asarray(accepted, dtype=bool)
    )
    accepted_n = int(accepted_mask.sum())
    accepted_errors = (
        int((pred[accepted_mask] != y[accepted_mask]).sum()) if accepted_n else 0
    )
    has_both_classes = np.unique(y).size == 2

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y[accepted_mask], pred[accepted_mask], labels=[0, 1], zero_division=0
        )
        if accepted_n
        else (np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    )

    cm = (
        confusion_matrix(y[accepted_mask], pred[accepted_mask], labels=[0, 1])
        if accepted_n
        else np.zeros((2, 2), dtype=int)
    )
    predicted_class_risk: dict[str, dict[str, float | int]] = {}
    for class_value, class_name in ((0, "TRANSIENT"), (1, "SRKW")):
        class_accepted = accepted_mask & (pred == class_value)
        class_n = int(class_accepted.sum())
        class_errors = int((y[class_accepted] != class_value).sum()) if class_n else 0
        predicted_class_risk[class_name] = {
            "accepted_n": class_n,
            "errors": class_errors,
            "error": float(class_errors / class_n) if class_n else np.nan,
            "error_upper": wilson_upper_bound(class_errors, class_n, confidence),
        }
    result: dict[str, float | int | dict] = {
        "N": int(len(y)),
        "SRKW_N": int((y == 1).sum()),
        "TRANSIENT_N": int((y == 0).sum()),
        "LOG_LOSS": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "BRIER": float(brier_score_loss(y, p)),
        # Ranking and balanced metrics are undefined for a one-class slice.
        # Small evidence-regime reports commonly have this shape, so report NaN
        # explicitly instead of emitting misleading sklearn warnings.
        "ROC_AUC": float(roc_auc_score(y, p)) if has_both_classes else np.nan,
        "AVERAGE_PRECISION": (
            float(average_precision_score(y, p)) if has_both_classes else np.nan
        ),
        "FORCED_ACCURACY": float(accuracy_score(y, pred)),
        "FORCED_BALANCED_ACCURACY": (
            float(balanced_accuracy_score(y, pred)) if has_both_classes else np.nan
        ),
        "ECE_10": expected_calibration_error(y, p, n_bins=10),
        "ACCEPTED_N": accepted_n,
        "COVERAGE": float(accepted_n / len(y)) if len(y) else 0.0,
        "SELECTIVE_ERRORS": accepted_errors,
        "SELECTIVE_ERROR": (
            float(accepted_errors / accepted_n) if accepted_n else np.nan
        ),
        "SELECTIVE_ACCURACY": (
            float(1 - accepted_errors / accepted_n) if accepted_n else np.nan
        ),
        "SELECTIVE_ERROR_UPPER": wilson_upper_bound(
            accepted_errors, accepted_n, confidence
        ),
        "CLASS_METRICS": {
            "TRANSIENT": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "SRKW": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "PREDICTED_CLASS_RISK": predicted_class_risk,
        "CONFUSION_MATRIX": {
            "labels": ["TRANSIENT", "SRKW"],
            "matrix": cm.astype(int).tolist(),
        },
    }
    return result


def metrics_by_regime(
    y: np.ndarray,
    p: np.ndarray,
    accepted: np.ndarray,
    regimes: np.ndarray,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for regime in sorted(pd.unique(regimes)):
        mask = np.asarray(regimes) == regime
        metrics = summarize_binary_metrics(
            np.asarray(y)[mask],
            np.asarray(p)[mask],
            accepted=np.asarray(accepted)[mask],
            confidence=confidence,
        )
        rows.append(
            {
                "EVIDENCE_REGIME": regime,
                "N": metrics["N"],
                "COVERAGE": metrics["COVERAGE"],
                "SELECTIVE_ACCURACY": metrics["SELECTIVE_ACCURACY"],
                "SELECTIVE_ERROR_UPPER": metrics["SELECTIVE_ERROR_UPPER"],
                "BRIER": metrics["BRIER"],
                "ROC_AUC": metrics["ROC_AUC"],
            }
        )
    return pd.DataFrame(rows)
