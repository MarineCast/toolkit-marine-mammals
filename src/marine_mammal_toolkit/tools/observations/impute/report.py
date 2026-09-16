from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages

from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_text

REPORT_METRICS = (
    "N",
    "BRIER",
    "LOG_LOSS",
    "ROC_AUC",
    "ECE_10",
    "COVERAGE",
    "SELECTIVE_ACCURACY",
    "SELECTIVE_ERROR_UPPER",
)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def metrics_table(evaluations: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for strategy, evaluation in evaluations.items():
        row = {"STRATEGY": strategy}
        row.update({name: evaluation.metrics.get(name) for name in REPORT_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def class_risk_table(evaluations: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for strategy, evaluation in evaluations.items():
        for predicted_class, risk in evaluation.metrics.get(
            "PREDICTED_CLASS_RISK", {}
        ).items():
            rows.append(
                {
                    "STRATEGY": strategy,
                    "PREDICTED_CLASS": predicted_class,
                    "ACCEPTED_N": risk.get("accepted_n"),
                    "ERRORS": risk.get("errors"),
                    "ERROR_RATE": risk.get("error"),
                    "ERROR_UPPER_95": risk.get("error_upper"),
                }
            )
    return pd.DataFrame(rows)


def certification_table(imputer: Any) -> pd.DataFrame:
    certification = pd.DataFrame.from_dict(
        imputer.training_summary_.get("CLASS_CERTIFICATION", {}), orient="index"
    ).reset_index(names="PREDICTED_CLASS")
    selected = [
        column
        for column in (
            "PREDICTED_CLASS",
            "CERTIFIED",
            "OUTER_ACCEPTED_N",
            "OUTER_ERRORS",
            "OUTER_ERROR_UPPER",
            "STRATEGY",
        )
        if column in certification
    ]
    return certification[selected]


def calibration_figure(reliability: pd.DataFrame, strategy: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#708090", "dash": "dash"},
        )
    )
    if not reliability.empty:
        figure.add_trace(
            go.Scatter(
                x=reliability["MEAN_P_SRKW"],
                y=reliability["OBSERVED_SRKW_RATE"],
                mode="lines+markers",
                name=f"{strategy.title()} validation",
                marker={"size": 9, "color": "#0072B2"},
                line={"color": "#0072B2", "width": 3},
                customdata=np.column_stack(
                    [reliability["COUNT"], reliability["LOWER"], reliability["UPPER"]]
                ),
                hovertemplate=(
                    "Mean P(SRKW): %{x:.3f}<br>Observed rate: %{y:.3f}"
                    "<br>N: %{customdata[0]:,.0f}<br>Bin: %{customdata[1]:.1f}-"
                    "%{customdata[2]:.1f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"Calibration curve - {strategy}",
        template="plotly_white",
        xaxis={"title": "Predicted P(SRKW)", "range": [0, 1]},
        yaxis={"title": "Observed SRKW rate", "range": [0, 1]},
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 65, "r": 25, "t": 90, "b": 60},
        height=520,
    )
    return figure


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='muted'>No data available.</p>"
    formatted = frame.copy()
    for column in formatted:
        formatted[column] = formatted[column].map(_fmt)
    return formatted.to_html(index=False, classes="metrics", border=0, escape=True)


def _metadata_table(metadata: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{html.escape(_fmt(value))}</td></tr>"
        for key, value in metadata.items()
    )
    return f"<table class='metadata'>{rows}</table>"


def _metric_value(metrics: dict[str, Any], name: str) -> float | int | None:
    value = metrics.get(name)
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_imputation_report_payload(
    *,
    imputer: Any,
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize model evidence and production attribution without conflating soft labels."""

    required = {
        "ECOTYPE_DETAIL_OBSERVED",
        "ECOTYPE_DETAIL_EFFECTIVE",
        "IMPUTATION_APPLIED",
        "P_SRKW",
        "P_TRANSIENT",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Imputation report is missing prediction columns: {missing}")

    summary = imputer.training_summary_
    strategy = str(
        summary.get("FINAL_CALIBRATION_STRATEGY")
        or imputer.config.model.final_calibration_strategy
    )
    if strategy not in imputer.evaluations_:
        raise ValueError(
            f"Imputation report cannot find evaluation strategy {strategy!r}"
        )
    selected_metrics = imputer.evaluations_[strategy].metrics
    raw_metrics = selected_metrics.get("RAW_PROBABILITY_METRICS", {})

    observed = predictions["ECOTYPE_DETAIL_OBSERVED"].fillna("UNKNOWN").astype(str)
    effective = predictions["ECOTYPE_DETAIL_EFFECTIVE"].fillna("UNKNOWN").astype(str)
    hard_applied = predictions["IMPUTATION_APPLIED"].fillna(False).astype(bool)
    unknown = observed.eq("UNKNOWN")
    probability_scored = unknown & predictions[["P_SRKW", "P_TRANSIENT"]].notna().any(
        axis=1
    )
    hard_counts = effective.loc[hard_applied].value_counts()
    hard_by_ecotype = {
        ecotype: int(hard_counts.get(ecotype, 0)) for ecotype in ("SRKW", "TRANSIENT")
    }
    for ecotype, count in hard_counts.items():
        if str(ecotype) not in hard_by_ecotype:
            hard_by_ecotype[str(ecotype)] = int(count)

    metric_names = ("N", "BRIER", "LOG_LOSS", "ROC_AUC", "ECE_10")
    calibrated = {
        name.lower(): _metric_value(selected_metrics, name) for name in metric_names
    }
    raw = {name.lower(): _metric_value(raw_metrics, name) for name in metric_names}
    model_config = imputer.config.model
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_fit": {
            "fit_run_id": summary.get("FIT_RUN_ID"),
            "fit_at_utc": summary.get("FIT_AT_UTC"),
            "model_version": summary.get("MODEL_VERSION"),
            "training_data_snapshot_id": summary.get("TRAINING_DATA_SNAPSHOT_ID"),
            "labeled_n": summary.get("LABELED_N"),
            "classifier_training_n": summary.get("CLASSIFIER_TRAINING_N"),
            "labeled_encounter_n": summary.get("LABELED_ENCOUNTER_N"),
            "feature_n": summary.get("FEATURE_N"),
            "validation_strategy": strategy,
            "selective_coverage": _metric_value(selected_metrics, "COVERAGE"),
            "selective_accuracy": _metric_value(selected_metrics, "SELECTIVE_ACCURACY"),
            "selective_error_upper": _metric_value(
                selected_metrics, "SELECTIVE_ERROR_UPPER"
            ),
        },
        "calibration": {
            "method": getattr(model_config, "calibration_method", None),
            "strategy": strategy,
            "raw": raw,
            "calibrated": calibrated,
        },
        "class_certification": summary.get("CLASS_CERTIFICATION", {}),
        "imputation": {
            "total_points": int(len(predictions)),
            "observed_labeled_points": int(observed.isin(["SRKW", "TRANSIENT"]).sum()),
            "unknown_query_points": int(unknown.sum()),
            "probability_scored_unknown_points": int(probability_scored.sum()),
            "hard_imputed_points": int(hard_applied.sum()),
            "hard_imputed_points_by_ecotype": hard_by_ecotype,
            "hard_abstained_unknown_points": int((unknown & ~hard_applied).sum()),
            "effective_ecotype_distribution": {
                str(label): int(count)
                for label, count in effective.value_counts().items()
            },
        },
        "artifact": metadata,
    }


def write_imputation_html(
    path: str | Path, *, payload: dict[str, Any], overwrite: bool = False
) -> Path:
    """Write the deliberately compact production imputation report."""

    output = Path(path)
    fit = payload["model_fit"]
    calibration = payload["calibration"]
    imputation = payload["imputation"]
    by_ecotype = imputation["hard_imputed_points_by_ecotype"]
    comparison = pd.DataFrame(
        [
            {
                "METRIC": name.upper(),
                "RAW": calibration["raw"].get(name),
                "CALIBRATED": calibration["calibrated"].get(name),
            }
            for name in ("brier", "log_loss", "roc_auc", "ece_10")
        ]
    )
    certification = pd.DataFrame.from_dict(
        payload.get("class_certification", {}), orient="index"
    ).reset_index(names="ECOTYPE")
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OrcaCast imputation report</title>
<style>
:root{{--navy:#082b50;--blue:#0072b2;--ink:#18324a;--muted:#60758a;--line:#dce5ec;--bg:#f4f7fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:var(--navy);color:white;padding:32px max(24px,calc((100vw - 920px)/2))}} header h1{{margin:0 0 5px;font-size:30px}} header p{{margin:0;opacity:.85}}
main{{max-width:920px;margin:24px auto;padding:0 20px 48px}} section{{background:white;border:1px solid var(--line);border-radius:12px;padding:21px;margin:16px 0}}
h2{{margin:0 0 13px;color:var(--navy)}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-radius:9px;padding:15px}} .card strong{{display:block;font-size:25px;color:var(--blue)}} .card span,.muted{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--navy);background:#edf5fa}} .metadata th{{background:white;width:44%}}
.note{{border-left:4px solid var(--blue);padding:10px 14px;background:#eef7fc}} @media print{{body{{background:white}} section{{break-inside:avoid}}}}
</style></head><body>
<header><h1>Imputation report</h1><p>Fit, calibration, and production attribution</p></header><main>
<section><h2>Production result</h2><div class="cards">
<div class="card"><strong>{_fmt(imputation['total_points'])}</strong><span>Total points</span></div>
<div class="card"><strong>{_fmt(imputation['hard_imputed_points'])}</strong><span>Hard imputed</span></div>
<div class="card"><strong>{_fmt(by_ecotype.get('SRKW', 0))}</strong><span>Hard SRKW</span></div>
<div class="card"><strong>{_fmt(by_ecotype.get('TRANSIENT', 0))}</strong><span>Hard Transient</span></div>
<div class="card"><strong>{_fmt(imputation['probability_scored_unknown_points'])}</strong><span>Probability-scored unknown</span></div>
</div><p class="note">Probability-scored points are reported separately. They are not counted as hard imputations unless <code>IMPUTATION_APPLIED=true</code>.</p></section>
<section><h2>Model fit</h2>{_metadata_table(fit)}</section>
<section><h2>Calibration</h2><p class="muted">Method: {html.escape(_fmt(calibration.get('method')))}; validation strategy: {html.escape(_fmt(calibration.get('strategy')))}.</p>{_html_table(comparison)}</section>
<section><h2>Hard-label certification</h2>{_html_table(certification)}</section>
<section><h2>Prediction scope</h2>{_metadata_table(imputation)}</section>
<section><h2>Artifact</h2>{_metadata_table(payload.get('artifact', {}))}</section>
</main></body></html>"""
    return atomic_write_text(output, body, overwrite=overwrite)


def write_training_html(
    path: str | Path,
    *,
    imputer: Any,
    metadata: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluations = imputer.evaluations_
    strategy = imputer.config.model.final_calibration_strategy
    selected = evaluations[strategy]
    plot = calibration_figure(selected.reliability, strategy).to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True},
    )
    summary = imputer.training_summary_
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OrcaCast sighting imputation fit report</title>
<style>
:root{{--navy:#082b50;--blue:#0072b2;--ink:#18324a;--muted:#60758a;--line:#dce5ec;--bg:#f4f7fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:linear-gradient(120deg,var(--navy),#125b8d);color:white;padding:42px max(28px,calc((100vw - 1120px)/2))}}
header h1{{margin:0 0 7px;font-size:34px}} header p{{margin:0;opacity:.86}}
main{{max-width:1120px;margin:28px auto;padding:0 22px 60px}} section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 4px 16px #082b5010}}
h2{{margin:0 0 16px;color:var(--navy)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}}
table{{width:100%;border-collapse:collapse}} th{{text-align:left;color:var(--navy);font-weight:650}} th,td{{padding:10px 11px;border-bottom:1px solid var(--line)}}
.metrics th{{background:#edf5fa;white-space:nowrap}} .metrics tr:hover td{{background:#f8fbfd}} .metadata th{{width:42%}} .muted{{color:var(--muted)}}
.callout{{border-left:5px solid var(--blue);background:#eef7fc;padding:14px 17px;border-radius:7px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}} @media print{{body{{background:white}} section{{box-shadow:none;break-inside:avoid}}}}
</style></head><body>
<header><h1>Sighting imputation fit report</h1><p>Calibrated selective ecotype attribution</p></header><main>
<section><h2>Fit overview</h2><div class="grid">{_metadata_table(metadata)}{_metadata_table({k: summary.get(k) for k in ('MODEL_VERSION', 'REGIME', 'LABELED_N', 'CLASSIFIER_TRAINING_N', 'LABELED_ENCOUNTER_N', 'FEATURE_N', 'FINAL_CALIBRATION_STRATEGY')})}</div></section>
<section><h2>Validation metrics</h2><p class="muted">Cross-fitted results. Hard labels are controlled by selective-error bounds, not forced accuracy alone.</p>{_html_table(metrics_table(evaluations))}</section>
<section><h2>Predicted-class risk</h2>{_html_table(class_risk_table(evaluations))}</section>
<section><h2>Production class certification</h2><div class="callout">A class is eligible for hard production assignments only when its outer-validation upper error bound meets the configured target.</div>{_html_table(certification_table(imputer))}</section>
<section><h2>Calibration</h2>{plot}</section>
<section><h2>Configuration</h2><pre><code>{html.escape(json.dumps(imputer.config.to_dict(), indent=2, default=str))}</code></pre></section>
</main></body></html>"""
    output.write_text(body, encoding="utf-8")
    return output


def _pdf_table(ax: Any, frame: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    ax.set_title(
        title, loc="left", fontsize=16, fontweight="bold", color="#082b50", pad=16
    )
    if frame.empty:
        ax.text(0, 0.9, "No data available", color="#60758a")
        return
    printable = frame.rename(
        columns={
            "SELECTIVE_ACCURACY": "SEL_ACCURACY",
            "SELECTIVE_ERROR_UPPER": "ERROR_UPPER",
            "PREDICTED_CLASS": "CLASS",
            "ERROR_UPPER_95": "ERROR_UPPER",
        }
    )
    values = [[_fmt(value) for value in row] for row in printable.to_numpy()]
    table = ax.table(
        cellText=values,
        colLabels=list(printable.columns),
        loc="upper left",
        cellLoc="left",
        bbox=[0, 0.02, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dce5ec")
        if row == 0:
            cell.set_facecolor("#e8f3f9")
            cell.set_text_props(weight="bold", color="#082b50")


def write_training_pdf(
    path: str | Path,
    *,
    imputer: Any,
    metadata: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    evaluations = imputer.evaluations_
    strategy = imputer.config.model.final_calibration_strategy
    selected = evaluations[strategy]
    with PdfPages(
        output, metadata={"Title": "OrcaCast sighting imputation fit report"}
    ) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(0.06, 0.91, "OrcaCast", fontsize=14, color="#0072b2", weight="bold")
        fig.text(
            0.06,
            0.84,
            "Sighting imputation fit report",
            fontsize=28,
            color="#082b50",
            weight="bold",
        )
        fig.text(
            0.06,
            0.79,
            "Calibrated selective ecotype attribution",
            fontsize=14,
            color="#60758a",
        )
        overview = {
            "FIT_AT_UTC": metadata.get("fit_at_utc"),
            "FIT_RUN_ID": metadata.get("fit_run_id"),
            "CONFIG_SHA256": metadata.get("config_sha256"),
            "MODEL_SHA256": metadata.get("model_sha256"),
            "MODEL_VERSION": imputer.training_summary_.get("MODEL_VERSION"),
            "LABELED_N": imputer.training_summary_.get("LABELED_N"),
            "CLASSIFIER_TRAINING_N": imputer.training_summary_.get(
                "CLASSIFIER_TRAINING_N"
            ),
            "FEATURE_N": imputer.training_summary_.get("FEATURE_N"),
            "FINAL_STRATEGY": imputer.training_summary_.get(
                "FINAL_CALIBRATION_STRATEGY"
            ),
            "PYTHON": metadata.get("python"),
            "SCIKIT_LEARN": metadata.get("scikit_learn"),
        }
        y = 0.69
        for key, value in overview.items():
            fig.text(
                0.07, y, str(key).replace("_", " ").title(), fontsize=9, color="#60758a"
            )
            fig.text(0.34, y, _fmt(value), fontsize=9.5, color="#18324a")
            y -= 0.045
        fig.text(
            0.06,
            0.08,
            f"Generated {datetime.now(timezone.utc).isoformat()}",
            fontsize=8,
            color="#60758a",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11.69, 8.27),
            gridspec_kw={"height_ratios": [1.15, 1]},
        )
        _pdf_table(axes[0], metrics_table(evaluations), "Validation metrics")
        _pdf_table(axes[1], class_risk_table(evaluations), "Predicted-class risk")
        fig.tight_layout(pad=2.2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        _pdf_table(ax, certification_table(imputer), "Production class certification")
        fig.tight_layout(pad=2.2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.plot([0, 1], [0, 1], "--", color="#708090", label="Perfect calibration")
        reliability = selected.reliability
        if not reliability.empty:
            ax.plot(
                reliability["MEAN_P_SRKW"],
                reliability["OBSERVED_SRKW_RATE"],
                "o-",
                color="#0072b2",
                linewidth=2.5,
                label=f"{strategy.title()} validation",
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Predicted P(SRKW)",
            ylabel="Observed SRKW rate",
            title=f"Calibration curve - {strategy}",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
        fig.tight_layout(pad=2.4)
        pdf.savefig(fig)
        plt.close(fig)
    return output
