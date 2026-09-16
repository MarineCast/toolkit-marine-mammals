from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAP_CLASSES = (
    "SRKW",
    "TRANSIENT",
    "SRKW_ASSIGNED",
    "TRANSIENT_ASSIGNED",
    "UNKNOWN_STILL",
)

MAP_COLORS = {
    "SRKW": "#0072B2",
    "TRANSIENT": "#D55E00",
    "SRKW_ASSIGNED": "#56B4E9",
    "TRANSIENT_ASSIGNED": "#E69F00",
    "UNKNOWN_STILL": "#777777",
    "KNOWN_OTHER": "#009E73",
}


def make_animated_map(
    frame: pd.DataFrame,
    *,
    year: int = 2025,
    frame_unit: str = "week",
    include_known_other: bool = False,
    zoom: float = 5.6,
    title: str | None = None,
) -> go.Figure:
    """Create an animated map of known, assigned, and abstained sightings."""

    required = {"SIGHTING_DATE", "LATITUDE", "LONGITUDE", "DISPLAY_CLASS"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Map input is missing columns: {missing}")
    data = frame.copy()
    data["SIGHTING_DATE"] = pd.to_datetime(
        data["SIGHTING_DATE"], format="mixed"
    ).dt.normalize()
    data = data.loc[data["SIGHTING_DATE"].dt.year.eq(year)].copy()
    allowed = set(MAP_CLASSES)
    if include_known_other:
        allowed.add("KNOWN_OTHER")
    data = data.loc[data["DISPLAY_CLASS"].isin(allowed)].copy()
    if data.empty:
        raise ValueError(f"No mappable sightings found for {year}")

    if frame_unit == "day":
        frame_date = data["SIGHTING_DATE"]
    elif frame_unit == "week":
        frame_date = data["SIGHTING_DATE"] - pd.to_timedelta(
            data["SIGHTING_DATE"].dt.weekday, unit="D"
        )
    elif frame_unit == "month":
        frame_date = data["SIGHTING_DATE"].dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError("frame_unit must be 'day', 'week', or 'month'")
    data["MAP_FRAME"] = frame_date.dt.strftime("%Y-%m-%d")
    data["DISPLAY_CLASS"] = pd.Categorical(
        data["DISPLAY_CLASS"],
        categories=[*MAP_CLASSES, "KNOWN_OTHER"],
        ordered=True,
    )
    data = data.sort_values(["MAP_FRAME", "DISPLAY_CLASS", "SIGHTING_DATE"])

    hover_columns: dict[str, str | bool] = {
        "OBSERVATION_ID": True,
        "SIGHTING_DATE": "|%Y-%m-%d",
        "DISPLAY_CLASS": False,
        "P_SRKW": ":.3f",
        "P_TRANSIENT": ":.3f",
        "EVIDENCE_REGIME": True,
        "ABSTENTION_REASON": True,
        "P_SRKW_RAW": ":.3f",
        "CONFORMAL_SET": True,
        "OOD_MARGIN": ":.3f",
        "NEAREST_SRKW_KM": ":.2f",
        "NEAREST_SRKW_DAY_LAG": True,
        "NEAREST_TRANSIENT_KM": ":.2f",
        "NEAREST_TRANSIENT_DAY_LAG": True,
        "DISTANCE_METHOD": True,
        "MARINE_LOOKUP_COVERAGE": ":.3f",
        "MARINE_ROUTED_NEIGHBORS": True,
        "MARINE_FALLBACK_NEIGHBORS": True,
        "MARINE_BARRIER_EXCLUDED_NEIGHBORS": True,
        "MARINE_TARGET_SNAP_KM": ":.2f",
        "MARINE_TARGET_KNOWN": True,
        "PAST_SRKW_SUPPORT": ":.3f",
        "PAST_TRANSIENT_SUPPORT": ":.3f",
        "FUTURE_SRKW_SUPPORT": ":.3f",
        "FUTURE_TRANSIENT_SUPPORT": ":.3f",
        "SOURCE_REPORT_COUNT": True,
        "LATITUDE": ":.4f",
        "LONGITUDE": ":.4f",
    }
    hover_columns = {
        key: value for key, value in hover_columns.items() if key in data.columns
    }
    size = "SOURCE_REPORT_COUNT" if "SOURCE_REPORT_COUNT" in data.columns else None
    figure = px.scatter_map(
        data,
        lat="LATITUDE",
        lon="LONGITUDE",
        color="DISPLAY_CLASS",
        color_discrete_map=MAP_COLORS,
        category_orders={"DISPLAY_CLASS": [*MAP_CLASSES, "KNOWN_OTHER"]},
        animation_frame="MAP_FRAME",
        hover_name="OBSERVATION_ID" if "OBSERVATION_ID" in data.columns else None,
        hover_data=hover_columns,
        size=size,
        size_max=13,
        zoom=zoom,
        center={
            "lat": float(data["LATITUDE"].median()),
            "lon": float(data["LONGITUDE"].median()),
        },
        map_style="open-street-map",
        title=title or f"Ecotype attribution over time, {year}",
        height=760,
    )
    figure.update_traces(marker={"opacity": 0.82})
    figure.update_layout(
        legend_title_text="Class",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    return figure


def save_map_html(figure: go.Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    return output


def plot_reliability(reliability: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    if not reliability.empty:
        ax.plot(
            reliability["MEAN_P_SRKW"],
            reliability["OBSERVED_SRKW_RATE"],
            marker="o",
            label="Cross-fitted model",
        )
    ax.set(
        xlabel="Predicted P(SRKW)",
        ylabel="Observed SRKW rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_risk_coverage(curve: pd.DataFrame, target_error: float | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = curve.dropna(subset=["SELECTIVE_ERROR"])
    ax.plot(
        valid["COVERAGE"], valid["SELECTIVE_ERROR"], label="Observed selective error"
    )
    ax.plot(valid["COVERAGE"], valid["ERROR_UPPER"], label="Upper confidence bound")
    if target_error is not None:
        ax.axhline(target_error, linestyle="--", linewidth=1, label="Target error")
    ax.set(xlabel="Coverage", ylabel="Error among accepted predictions", ylim=(0, 1))
    ax.set_title("Risk–coverage curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(metrics: dict):
    labels = metrics["CONFUSION_MATRIX"]["labels"]
    matrix = np.asarray(metrics["CONFUSION_MATRIX"]["matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Accepted-prediction confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
