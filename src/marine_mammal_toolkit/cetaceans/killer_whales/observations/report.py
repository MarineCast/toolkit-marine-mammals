"""Build a portable HTML summary of the killer-whale sightings product."""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.parquet as pq

from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_text

DATE_COLUMN = "SIGHTING_DATE_UTC"
ID_COLUMN = "OBSERVATION_ID"
MAP_COLUMNS = (ID_COLUMN, DATE_COLUMN, "LATITUDE", "LONGITUDE", "SOURCE")
REPORT_SCHEMA_VERSION = "1"


def _load_composite(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path)
    missing = sorted(set(MAP_COLUMNS) - set(schema.names))
    if missing:
        raise ValueError(f"Composite sightings are missing report columns: {missing}")
    frame = pq.read_table(path, columns=list(MAP_COLUMNS)).to_pandas()
    if frame.empty:
        raise ValueError(
            "Cannot build a sightings report from an empty composite table"
        )
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"Composite sightings require unique, non-null {ID_COLUMN}")
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], utc=True, errors="coerce")
    if frame[DATE_COLUMN].isna().any():
        raise ValueError(f"Composite sightings contain invalid {DATE_COLUMN} values")
    frame["LATITUDE"] = pd.to_numeric(frame["LATITUDE"], errors="coerce")
    frame["LONGITUDE"] = pd.to_numeric(frame["LONGITUDE"], errors="coerce")
    frame["SOURCE"] = frame["SOURCE"].astype("string").fillna("UNKNOWN")
    return frame


def _daily_counts(frame: pd.DataFrame) -> pd.DataFrame:
    observed = frame[DATE_COLUMN].dt.floor("D").value_counts().sort_index()
    calendar = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    counts = observed.reindex(calendar, fill_value=0).rename("REPORTED_SIGHTINGS")
    result = counts.rename_axis("DATE").reset_index()
    result["ROLLING_30_DAY_MEAN"] = (
        result["REPORTED_SIGHTINGS"].rolling(30, min_periods=1).mean()
    )
    return result


def _source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby("SOURCE", dropna=False)
        .agg(
            SIGHTINGS=(ID_COLUMN, "size"),
            FIRST_SIGHTING=(DATE_COLUMN, "min"),
            LAST_SIGHTING=(DATE_COLUMN, "max"),
        )
        .reset_index()
        .sort_values(["SIGHTINGS", "SOURCE"], ascending=[False, True])
    )
    for column in ("FIRST_SIGHTING", "LAST_SIGHTING"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def _map_zoom(frame: pd.DataFrame) -> float:
    latitude_span = max(float(frame["LATITUDE"].max() - frame["LATITUDE"].min()), 0.1)
    longitude_span = max(
        float(frame["LONGITUDE"].max() - frame["LONGITUDE"].min()), 0.1
    )
    span = max(latitude_span, longitude_span * 0.65)
    return max(1.0, min(7.0, 7.2 - math.log2(span)))


def _density_cells(frame: pd.DataFrame, *, h3_resolution: int) -> pd.DataFrame:
    mappable = frame.loc[
        frame["LATITUDE"].between(-90, 90) & frame["LONGITUDE"].between(-180, 180)
    ].dropna(subset=["LATITUDE", "LONGITUDE"])
    if mappable.empty:
        raise ValueError(
            "Composite sightings contain no valid coordinates for the density map"
        )
    cells = [
        h3.latlng_to_cell(float(latitude), float(longitude), h3_resolution)
        for latitude, longitude in zip(
            mappable["LATITUDE"], mappable["LONGITUDE"], strict=True
        )
    ]
    counts = pd.Series(cells, name="H3_CELL").value_counts().rename("SIGHTINGS")
    result = counts.rename_axis("H3_CELL").reset_index()
    centers = result["H3_CELL"].map(h3.cell_to_latlng)
    result["LATITUDE"] = centers.map(lambda value: value[0])
    result["LONGITUDE"] = centers.map(lambda value: value[1])
    return result


def _density_figure(cells: pd.DataFrame, mappable: pd.DataFrame) -> go.Figure:
    figure = px.density_map(
        cells,
        lat="LATITUDE",
        lon="LONGITUDE",
        z="SIGHTINGS",
        hover_name="H3_CELL",
        hover_data={
            "SIGHTINGS": ":,",
            "LATITUDE": ":.3f",
            "LONGITUDE": ":.3f",
        },
        color_continuous_scale=[
            [0.0, "#d8edf5"],
            [0.2, "#72c5d7"],
            [0.5, "#1585a5"],
            [0.75, "#155080"],
            [1.0, "#ff8a3d"],
        ],
        radius=22,
        zoom=_map_zoom(mappable),
        center={
            "lat": float(mappable["LATITUDE"].median()),
            "lon": float(mappable["LONGITUDE"].median()),
        },
        map_style="open-street-map",
        title="All-time density of reported killer-whale sightings",
        height=690,
    )
    figure.update_layout(
        coloraxis_colorbar={"title": "Sightings"},
        margin={"l": 0, "r": 0, "t": 56, "b": 0},
    )
    return figure


def _timeline_figure(daily: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=daily["DATE"],
            y=daily["REPORTED_SIGHTINGS"],
            mode="lines",
            line={"color": "#8fc7d8", "width": 1},
            name="Daily count",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} sightings<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=daily["DATE"],
            y=daily["ROLLING_30_DAY_MEAN"],
            mode="lines",
            line={"color": "#0b5679", "width": 2.5},
            name="30-day rolling mean",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} mean sightings<extra></extra>",
        )
    )
    figure.update_layout(
        title="Reported sightings count over time",
        xaxis_title="Sighting date",
        yaxis_title="Sightings in composite table",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        margin={"l": 62, "r": 20, "t": 72, "b": 58},
    )
    figure.update_xaxes(rangeslider_visible=True)
    return figure


def _imputation_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    schema = pq.read_schema(path)
    if "IMPUTATION_APPLIED" not in schema.names:
        return None
    column = pq.read_table(path, columns=["IMPUTATION_APPLIED"])["IMPUTATION_APPLIED"]
    return sum(value is True for value in column.to_pylist())


def _manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sightings model manifest is not an object: {path}")
    return payload


def build_sightings_report_html(
    *,
    composite_path: str | Path,
    imputed_path: str | Path,
    model_manifest_path: str | Path,
    output_path: str | Path,
    h3_resolution: int = 6,
    overwrite: bool = False,
    display_root: str | Path | None = None,
) -> Path:
    """Build the consumer-facing all-time sightings density and timeline report."""

    composite = Path(composite_path).expanduser().resolve()
    imputed = Path(imputed_path).expanduser().resolve()
    manifest_path = Path(model_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    display_composite = (
        Path(display_root) / composite.name if display_root else composite
    )
    display_imputed = Path(display_root) / imputed.name if display_root else imputed
    display_manifest = (
        Path(display_root) / manifest_path.name if display_root else manifest_path
    )
    if not 0 <= h3_resolution <= 15:
        raise ValueError("h3_resolution must be between 0 and 15")

    sightings = _load_composite(composite)
    mappable = sightings.loc[
        sightings["LATITUDE"].between(-90, 90)
        & sightings["LONGITUDE"].between(-180, 180)
    ].dropna(subset=["LATITUDE", "LONGITUDE"])
    density = _density_cells(sightings, h3_resolution=h3_resolution)
    daily = _daily_counts(sightings)
    sources = _source_summary(sightings)
    manifest = _manifest(manifest_path)
    hard_imputed = _imputation_count(imputed)

    map_html = _density_figure(density, mappable).to_html(
        full_html=False,
        include_plotlyjs="inline",
        div_id="reported-sightings-density",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    timeline_html = _timeline_figure(daily).to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="reported-sightings-timeline",
        config={"displaylogo": False, "responsive": True},
    )
    start = sightings[DATE_COLUMN].min().strftime("%Y-%m-%d")
    end = sightings[DATE_COLUMN].max().strftime("%Y-%m-%d")
    excluded_coordinates = len(sightings) - len(mappable)
    release_id = str(manifest.get("release_id") or "unavailable")
    public_eligible = manifest.get("public_eligible") is True
    generated_at = datetime.now(timezone.utc).isoformat()
    imputation_display = "Unavailable" if hard_imputed is None else f"{hard_imputed:,}"
    source_display = sources.rename(
        columns={
            "SOURCE": "Source",
            "SIGHTINGS": "Sightings",
            "FIRST_SIGHTING": "First sighting",
            "LAST_SIGHTING": "Last sighting",
        }
    )
    source_display["Sightings"] = source_display["Sightings"].map(
        lambda value: f"{value:,}"
    )
    source_table = source_display.to_html(
        index=False, classes="source-table", border=0, escape=True
    )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Killer-whale sightings report</title>
<style>
:root{{--navy:#082b50;--ocean:#0b7895;--ink:#193449;--muted:#63788a;--line:#d9e4ea;--bg:#f3f7f9;--warm:#ff8a3d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:linear-gradient(125deg,var(--navy),#0b6685);color:white;padding:42px max(24px,calc((100vw - 1180px)/2))}}
header h1{{margin:0 0 6px;font-size:36px}} header p{{margin:0;opacity:.88}} main{{max-width:1180px;margin:26px auto;padding:0 22px 60px}}
section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 4px 18px #082b500d}}
h2{{margin:0 0 12px;color:var(--navy)}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:15px}} .card strong{{display:block;font-size:25px;color:var(--ocean)}} .card span,.muted{{color:var(--muted)}}
.note{{border-left:4px solid var(--warm);padding:11px 14px;background:#fff6ef}} .plot{{width:100%;overflow:hidden}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}} th{{background:#edf5f8;color:var(--navy)}}
.metadata{{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}} .status{{font-weight:700;color:{'#237447' if public_eligible else '#a34718'}}}
@media (max-width:680px){{header h1{{font-size:29px}} section{{padding:16px}}}} @media print{{body{{background:white}} section{{break-inside:avoid;box-shadow:none}}}}
</style></head><body>
<header><h1>Killer-whale sightings report</h1><p>Composite observations and selective ecotype-label imputation</p></header><main>
<section><h2>Product summary</h2><div class="cards">
<div class="card"><strong>{len(sightings):,}</strong><span>Composite sightings</span></div>
<div class="card"><strong>{len(mappable):,}</strong><span>Mappable sightings</span></div>
<div class="card"><strong>{len(sources):,}</strong><span>Sources</span></div>
<div class="card"><strong>{imputation_display}</strong><span>Hard-imputed labels</span></div>
</div><p class="muted">Observed from {start} through {end}. Generated {html.escape(generated_at)}.</p>
<p class="note">These figures describe reported records in the composite table. They do not estimate whale abundance, occupancy, survey effort, reporting probability, or verified absence. Unknown and abstained labels remain distinct from zero.</p></section>
<section><h2>All-time reported-sighting density</h2><p class="muted">All {len(mappable):,} valid coordinates are counted and aggregated to H3 resolution {h3_resolution} before display; {excluded_coordinates:,} records with invalid or unavailable coordinates are excluded. Map tiles require an internet connection when the report is opened.</p><div class="plot">{map_html}</div></section>
<section><h2>Sources</h2>{source_table}</section>
<section><h2>Product identity</h2><p class="metadata"><strong>Release:</strong> {html.escape(release_id)}<br><strong>Report schema:</strong> {REPORT_SCHEMA_VERSION}<br><strong>Public eligible:</strong> <span class="status">{'yes' if public_eligible else 'no'}</span><br><strong>Composite:</strong> {html.escape(str(display_composite))}<br><strong>Imputed:</strong> {html.escape(str(display_imputed))}<br><strong>Model manifest:</strong> {html.escape(str(display_manifest))}</p></section>
<section><h2>Reported sightings count over time</h2><p class="muted">Daily row counts and a 30-day rolling mean. A zero means the composite table contains no record for that date; it is not evidence of survey coverage or whale absence.</p><div class="plot">{timeline_html}</div></section>
</main></body></html>"""
    return atomic_write_text(output, body, overwrite=overwrite)
