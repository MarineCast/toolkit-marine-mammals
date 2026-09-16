"""Build the CWR exploration report from explicit tables, figures, and provenance.

No notebook globals, acquisition, or repository discovery are used on import.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import pandas as pd

def build_report(*,
    ARCHIVE_INDEX_URL,
    ARCHIVE_YEARS,
    REPORT_PATH,
    archive,
    archive_cache,
    archive_retrieved_at_utc,
    atlist,
    atlist_pull_times,
    comparison_snapshot_created_at,
    comparison_snapshot_id,
    comparison_snapshot_sources,
    comparison_source_report,
    comparison_status_report,
    comparison_year_report,
    cwr_reference_rows_excluded,
    ecotype_counts,
    encounter_map,
    encounters,
    high_priority_comparison,
    high_priority_count,
    identity_supported_count,
    latest_in_year_month,
    mapped,
    nearby_count,
    no_close_evidence_count,
    orcacast_observations_all,
    series_months,
    time_series_fig,
    very_close_count,
    year_ecotype_counts,
):
    def dataframe_html(frame: pd.DataFrame, index: bool = False) -> str:
        return frame.to_html(index=index, border=0, classes="data-table", escape=True)


    report_generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    source_pull_times = sorted(set([archive_retrieved_at_utc, *atlist_pull_times]))
    latest_source_pull_at_utc = max(source_pull_times)
    year_report = year_ecotype_counts.reset_index().rename(columns={"source_year": "Source year"})
    ecotype_report = ecotype_counts.rename(columns={"ecotype_display": "Ecotype", "encounters": "Encounters"})

    coverage_report = (
        encounters.assign(mapped=encounters[["map_lat", "map_lon"]].notna().all(axis=1))
        .groupby(["source_year", "source_system"], as_index=False)
        .agg(Encounters=("source_record_key", "count"), Mapped=("mapped", "sum"))
        .rename(columns={"source_year": "Source year", "source_system": "Source system"})
    )
    coverage_report["Mapped"] = coverage_report["Mapped"].astype(int)
    coverage_report["Map coverage"] = coverage_report.apply(
        lambda row: f"{100 * row['Mapped'] / row['Encounters']:.1f}%", axis=1
    )

    archive_inventory_rows = []
    for year in ARCHIVE_YEARS:
        year_archive = archive.loc[archive["source_year"].eq(year)]
        index_info = archive_cache["index_results"][str(year)]
        archive_inventory_rows.append({
            "Year": year,
            "Index entries": index_info["entry_count"],
            "Canonical encounters": len(year_archive),
            "Standard": int(year_archive["record_series"].eq("encounter").sum()),
            "UAV": int(year_archive["record_series"].eq("uav_encounter").sum()),
            "Mapped": int(year_archive[["map_lat", "map_lon"]].notna().all(axis=1).sum()),
        })
    archive_inventory_report = pd.DataFrame(archive_inventory_rows)

    qc_counter = Counter(flag for flags in encounters["qc_flags"] for flag in flags)
    qc_summary = pd.DataFrame(
        [{"QC flag": flag, "Encounter records": count} for flag, count in qc_counter.most_common()]
    )
    qc_rows = encounters.loc[
        encounters["qc_flags"].map(bool),
        ["source_year", "record_series", "encounter_number", "source_record_name", "qc_flags"],
    ].copy()
    qc_rows["qc_flags"] = qc_rows["qc_flags"].map(lambda values: ", ".join(values))

    time_series_html = time_series_fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True},
    )
    map_document = encounter_map.get_root().render()
    map_srcdoc = html.escape(map_document, quote=True)

    metric_cards = "".join(
        f'<article class="metric"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></article>'
        for label, value in [
            ("CWR encounter records", f"{len(encounters):,}"),
            ("Mapped records", f"{len(mapped):,} of {len(encounters):,}"),
            ("2024–2026 map coverage", f"{int(atlist[['map_lat', 'map_lon']].notna().all(axis=1).sum()):,} of {len(atlist):,}"),
            ("Historical coverage", "2017–2026"),
            ("Latest source pull (UTC)", latest_source_pull_at_utc),
        ]
    )
    comparison_metric_cards = "".join(
        f'<article class="metric"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></article>'
        for label, value in [
            ("Identity-supported", f"{identity_supported_count:,}"),
            ("Same day + within 500 m", f"{very_close_count:,}"),
            ("Same day + 0.5–5 km", f"{nearby_count:,}"),
            ("No close/evaluable evidence", f"{no_close_evidence_count:,}"),
        ]
    )

    report_html = f'''<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CWR Encounter Data: Internal Sightings Stream</title>
    <style>
    :root {{ --ink:#102A43; --muted:#52667A; --line:#D9E2EC; --panel:#F6F9FC; --accent:#0072B2; --green:#0F766E; --amber:#B45309; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:#EDF3F8; font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1220px, calc(100% - 32px)); margin:32px auto 64px; }}
    header, section {{ background:white; border:1px solid var(--line); border-radius:14px; box-shadow:0 8px 24px rgba(16,42,67,.06); }}
    header {{ padding:32px; border-top:5px solid var(--accent); }}
    section {{ margin-top:20px; padding:26px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(29px,4vw,44px); line-height:1.08; letter-spacing:-.025em; }}
    h2 {{ margin:0 0 14px; font-size:23px; line-height:1.2; }}
    h3 {{ margin:24px 0 10px; font-size:17px; }}
    p {{ max-width:92ch; }}
    ul, ol {{ max-width:90ch; }}
    li + li {{ margin-top:7px; }}
    .lede {{ color:var(--muted); font-size:18px; margin:0; }}
    .eyebrow {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.08em; margin:0 0 8px; text-transform:uppercase; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-top:24px; }}
    .metric {{ padding:15px 16px; background:var(--panel); border:1px solid var(--line); border-radius:10px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:5px; font-size:19px; line-height:1.25; overflow-wrap:anywhere; }}
    .decision-grid, .two-col {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; align-items:start; }}
    .decision {{ border:1px solid var(--line); border-radius:10px; padding:17px; background:var(--panel); min-height:100%; }}
    .decision strong {{ display:block; color:var(--green); font-size:18px; margin-bottom:4px; }}
    .decision.caution strong {{ color:var(--amber); }}
    .bottom-line {{ border-top:5px solid var(--green); }}
    .answer {{ border-top:5px solid var(--accent); }}
    .scroll {{ overflow-x:auto; }}
    .data-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .data-table th, .data-table td {{ padding:9px 10px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }}
    .data-table th {{ background:var(--panel); font-weight:700; white-space:nowrap; }}
    .chart {{ min-height:500px; }}
    .map-frame {{ width:100%; min-height:700px; border:1px solid var(--line); border-radius:10px; background:#E9F1F7; }}
    .callout {{ padding:14px 16px; border-left:4px solid var(--accent); background:#EDF7FC; border-radius:6px; color:#234E67; }}
    .success {{ border-left-color:var(--green); background:#ECFDF5; color:#115E59; }}
    .warning {{ border-left-color:var(--amber); background:#FFF7ED; color:#7C2D12; }}
    .architecture {{ display:grid; gap:12px; margin:20px 0 8px; }}
    .architecture-path {{ display:grid; grid-template-columns:minmax(160px,1fr) 36px minmax(190px,1.2fr) 36px minmax(190px,1.2fr); gap:8px; align-items:center; }}
    .architecture-node {{ padding:15px; border:1px solid #B8CBDC; border-radius:9px; background:var(--panel); text-align:center; }}
    .architecture-node strong {{ display:block; }}
    .architecture-arrow {{ color:var(--accent); font-size:28px; font-weight:800; text-align:center; }}
    .architecture-merge {{ justify-self:center; width:min(680px,100%); padding:16px; border:2px solid var(--accent); border-radius:10px; background:#EDF7FC; text-align:center; }}
    .architecture-down {{ color:var(--accent); font-size:28px; line-height:1; text-align:center; }}
    details {{ margin-top:14px; border:1px solid var(--line); border-radius:9px; background:#FBFCFE; padding:0 16px 16px; }}
    summary {{ cursor:pointer; font-weight:750; padding:14px 0; }}
    .fine-print {{ color:var(--muted); font-size:13px; }}
    a {{ color:#005A8D; }}
    code {{ background:#EEF2F6; padding:.1em .3em; border-radius:4px; }}
    footer {{ color:var(--muted); margin-top:20px; font-size:13px; text-align:center; }}
    @media (max-width:800px) {{
      .decision-grid, .two-col {{ grid-template-columns:1fr; }}
      .architecture-path {{ grid-template-columns:1fr; }}
      .architecture-arrow {{ transform:rotate(90deg); }}
      header, section {{ padding:20px; }}
      .map-frame {{ min-height:540px; }}
    }}
    </style>
    </head>
    <body>
    <main>
    <header>
      <p class="eyebrow">Center for Whale Research data-source assessment</p>
      <h1>CWR Encounter Data: Internal Sightings Stream</h1>
      <p class="lede">Collection status, historical coverage, and overlap with other sightings sources</p>
      <div class="metrics">{metric_cards}</div>
    </header>

    <section class="bottom-line">
      <h2>Bottom line</h2>
      <p class="callout success"><strong>CWR is now a configured internal sightings stream.</strong> The extraction produced <strong>{len(encounters):,}</strong> encounter records, of which <strong>{len(mapped):,}</strong> are map-ready; all <strong>{len(atlist):,}</strong> records from 2024–2026 have coordinates. The overlap screen explicitly removes every CWR-bearing canonical observation from its reference set before matching.</p>
      <p>Some CWR encounters have plausible counterparts in other sources—most notably Maplify among the strongest matches—but most do not have strong duplication evidence under this screen. CWR may contribute to internal modeling and counts. It remains ineligible for public point release until permission and redistribution terms are confirmed.</p>
    </section>

    <section class="answer">
      <h2>Answer to the original question</h2>
      <div class="decision-grid">
        <article class="decision"><strong>Configured internal stream</strong>Public archive pages and unauthenticated Atlist JSON endpoints feed the maintained collection and normalization pipeline.</article>
        <article class="decision caution"><strong>Public release remains closed</strong>The source license is unknown, so CWR-only and CWR-merged observations are internal-only until permission is confirmed.</article>
        <article class="decision"><strong>Indirect overlap is possible</strong>Maplify and other channels can describe the same real-world whale event even when records have different source IDs, coordinates, or wording.</article>
        <article class="decision"><strong>Probably additive, pending deduplication</strong>Only {high_priority_count:,} of {len(encounters):,} records are high-priority overlap candidates. The other {len(encounters) - high_priority_count:,} are not necessarily unique, but they lack comparably strong evidence in this screen.</article>
      </div>
    </section>

    <section id="extraction">
      <h2>How the data were extracted</h2>
      <div class="two-col">
        <div>
          <h3>2017–2023: Wix archive pages</h3>
          <ol>
            <li>Open the CWR archive landing page and follow each exposed yearly index for 2017–2023.</li>
            <li>Read every encounter entry and its linked detail page; no separate pagination API was identified.</li>
            <li>Parse labeled fields such as date, observation times, pods, location, and start/end coordinates.</li>
            <li>Aggregate component pages for multi-sequence encounters while retaining standard and UAV series as distinct records.</li>
            <li>Retain source URLs, page and index checksums, original coordinate text, parse method, and QC flags.</li>
          </ol>
          <p class="fine-print">Result: {len(archive):,} canonical archive encounters; {int(archive[['map_lat', 'map_lon']].notna().all(axis=1).sum()):,} map-ready.</p>
        </div>
        <div>
          <h3>2024–2026: Atlist map JSON</h3>
          <ol>
            <li>Use each public CWR Atlist map ID for 2024, 2025, and 2026.</li>
            <li>Issue unauthenticated GET requests to <code>/v1/map/{{map_id}}/fields</code> and <code>/v1/map/{{map_id}}/markers</code>.</li>
            <li>Read each JSON marker's ID, title, tags, notes, latitude/longitude, and created/updated timestamps.</li>
            <li>Parse labeled notes for encounter summary, observation times, vessel, staff, observers, pods, IDs, and location description.</li>
            <li>Exclude non-encounter pins, harmonize the 2024 tag labels, and preserve response checksums and map update times.</li>
          </ol>
          <p class="fine-print">The marker point is a source-reported map coordinate; its role as encounter start, end, or another reference point is not stated. Result: {len(atlist):,} encounters, all map-ready.</p>
        </div>
      </div>
      <h3>Source architecture</h3>
      <div class="architecture" role="img" aria-label="Two extraction paths merge into one normalized CWR encounter dataset and its analytical outputs">
        <div class="architecture-path">
          <div class="architecture-node"><strong>2017–2023</strong>Wix archive index</div><div class="architecture-arrow">→</div><div class="architecture-node"><strong>Encounter detail pages</strong>Labeled fields + coordinate text</div><div class="architecture-arrow">→</div><div class="architecture-node"><strong>Archive parser</strong>Aggregation + coordinate QC</div>
        </div>
        <div class="architecture-path">
          <div class="architecture-node"><strong>2024–2026</strong>Public Atlist maps</div><div class="architecture-arrow">→</div><div class="architecture-node"><strong>Fields + markers JSON</strong>IDs, notes, tags, marker coordinates</div><div class="architecture-arrow">→</div><div class="architecture-node"><strong>Atlist parser</strong>Note labels + tag harmonization</div>
        </div>
        <div class="architecture-down">↓</div>
        <div class="architecture-merge"><strong>Normalized CWR encounter dataset</strong><br>{len(encounters):,} records → CSV, map, time series, metrics, QC, and overlap screen</div>
      </div>
    </section>

    <section id="comparison">
      <h2>Comparison with a separate local sightings dataset</h2>
      <p><strong>What was compared:</strong> a local normalized research snapshot containing {len(orcacast_observations_all):,} canonical observations after excluding {cwr_reference_rows_excluded:,} CWR-bearing observations. The reference includes Acartia, curated GBIF, iNaturalist, Maplify/WASEAK, and a Whale Museum extract. It is <strong>not</strong> a SalishSea.io export, so these results do not directly measure what SalishSea.io already holds.</p>
      <p class="callout"><strong>Overlap screen:</strong> {identity_supported_count:,} identity-supported candidates, {very_close_count:,} additional same-day candidates within 500 m, {nearby_count:,} weaker same-day candidates 0.5–5 km away, and {no_close_evidence_count:,} records with no close or evaluable evidence. Only <strong>{high_priority_count:,} of {len(encounters):,}</strong> are high-priority overlap candidates. The remaining <strong>{len(encounters) - high_priority_count:,}</strong> are not necessarily unique; this screen only says that equally strong duplication evidence was not found.</p>
      <div class="metrics">{comparison_metric_cards}</div>
      <p class="callout warning"><strong>Maplify deserves specific attention:</strong> {int(high_priority_comparison['existing_preferred_source'].eq('MAPLIFY').sum()):,} of the {high_priority_count:,} strongest candidates prefer a Maplify record. This supports a meaningful indirect-overlap hypothesis, but it does not prove that Maplify copied CWR or that the two records share a source lineage.</p>
      <div class="two-col">
        <div class="scroll"><h3>Screening result</h3>{dataframe_html(comparison_status_report)}</div>
        <div class="scroll"><h3>Preferred source for high-priority candidates</h3>{dataframe_html(comparison_source_report)}</div>
      </div>
      <details><summary>Detailed overlap method and results by CWR year</summary>
        <div class="scroll">{dataframe_html(comparison_year_report)}</div>
        <p>Reference snapshot <code>{html.escape(comparison_snapshot_id)}</code>, built {html.escape(comparison_snapshot_created_at)}, with source watermarks for {html.escape(', '.join(comparison_snapshot_sources))}.</p>
        <ul>
          <li><strong>Identity-supported:</strong> exact calendar date plus at least one shared named whale or Bigg's social-group identifier. Broad J/K/L pod and ecotype labels alone are not identity evidence.</li>
          <li><strong>Very close:</strong> exact date, no conflicting known ecotype, and closest existing coordinate within 500 m.</li>
          <li><strong>Nearby:</strong> the same test at 0.5–5 km. This is possible overlap, not confirmation.</li>
          <li><code>UNKNOWN</code> or <code>MIXED</code> ecotypes are non-conflicting; known contradictory ecotypes are excluded.</li>
        </ul>
      </details>
    </section>

    <section id="salishsea">
      <h2>How this relates to SalishSea.io</h2>
      <p>SalishSea.io's public occurrence display distinguishes a <strong>collection</strong>—the original observation channel—from a <strong>provider</strong>—the route by which a record was added to the service. That distinction matters: an Orca Network observation, for example, can arrive through a Maplify/Conserve pathway rather than through a direct Orca Network feed.</p>
      <p>No direct CWR provider was identified in the public application reviewed on 2026-08-19. The site does display records delivered through Maplify and publicly describes community sources that include Orca Network, Whale Alert, and HappyWhale. Because provider and collection can differ, some CWR encounters could still correspond indirectly to events already visible there.</p>
      <p class="callout warning"><strong>Interpretation limit:</strong> this report did not export and compare the full SalishSea.io occurrence table. The direct comparison above is against the separate local reference snapshot with CWR-bearing observations removed, and should not be described as a SalishSea.io deduplication result.</p>
      <p><a href="https://salishsea.io/">Open SalishSea.io</a></p>
    </section>

    <section id="recommendation">
      <h2>Recommendation</h2>
      <p class="callout success"><strong>Keep CWR enabled for internal use and fail closed for public point release.</strong></p>
      <ol>
        <li>Confirm CWR permission and redistribution terms for structured fields, coordinates, identifiers, and derived outputs.</li>
        <li>Ask whether CWR can provide an official feed, export, or stable API rather than relying indefinitely on public-page structure.</li>
        <li>Preserve CWR encounter IDs, source URLs, page or response checksums, pull timestamps, and field-level provenance.</li>
        <li>Continue running records through the canonical identity and duplicate resolver, using whale and social-group identifiers as well as date, space, and time.</li>
        <li>Measure genuinely additive observations only after canonical resolution; do not treat all {len(encounters) - high_priority_count:,} non-high-priority records as new.</li>
      </ol>
    </section>

    <section id="overview">
      <h2>Dataset overview</h2>
      <p>The CWR archive navigation exposes 2017–2023, and the current public maps cover 2024–2026. Earlier years are unavailable through the linked archive navigation rather than counted as zero.</p>
      <div class="two-col">
        <div class="scroll"><h3>Total sightings by ecotype</h3>{dataframe_html(ecotype_report)}</div>
        <div class="scroll"><h3>Source-year totals</h3>{dataframe_html(year_report)}</div>
      </div>
      <p class="callout warning"><strong>Modeling caveat:</strong> these are positive research encounters, not presence/absence observations and not effort-corrected occurrence rates. A day with no CWR encounter record must not be interpreted as evidence that whales were absent.</p>
    </section>

    <section id="map">
      <h2>Encounter map</h2>
      <p>Archive records use source-reported start coordinates where available and end coordinates only as an explicit fallback. Unsigned archive longitudes are interpreted as west within the study domain. Candidates outside the broad 45–52°N, 120–130°W plausibility gate retain their source coordinate text and QC flag but are withheld from the map. Atlist records retain their map-marker coordinates.</p>
      <iframe class="map-frame" title="CWR 2017–2026 encounter map" srcdoc="{map_srcdoc}"></iframe>
    </section>

    <section id="temporal">
      <h2>Time series of sightings by ecotype</h2>
      <p>Monthly zeroes are filled from January 2017 through {latest_in_year_month.strftime('%B %Y')}, the latest in-year 2026 encounter month in the source. Later 2026 months are not represented as zero.</p>
      <div class="chart">{time_series_html}</div>
    </section>

    <section id="qc-provenance">
      <h2>QC and provenance</h2>
      <p>Archive pull: <strong>{html.escape(archive_retrieved_at_utc)}</strong>. Atlist pull(s): <strong>{html.escape(', '.join(atlist_pull_times))}</strong>. Report generated: <strong>{html.escape(report_generated_at_utc)}</strong>.</p>
      <p>Blank fields mean the source did not state or expose a value. Time zones are not inferred. Full archive narratives and images are not copied into this dataset.</p>
      <details><summary>Map coverage by source year</summary><div class="scroll">{dataframe_html(coverage_report)}</div></details>
      <details><summary>Archive inventory</summary><div class="scroll">{dataframe_html(archive_inventory_report)}</div></details>
      <details><summary>QC summary and flagged records ({len(qc_rows):,})</summary>
        <div class="scroll">{dataframe_html(qc_summary) if len(qc_summary) else '<p>None.</p>'}</div>
        <div class="scroll">{dataframe_html(qc_rows)}</div>
        <p class="callout warning">One 2020 archive page URL is reused by encounter entries #1 and #2. The index identity is retained for both, but page-level fields are withheld from the mismatched entry. One 2017 <code>CA/U</code> label remains unknown rather than being assigned an ecotype.</p>
      </details>
    </section>

    <section id="detailed-method">
      <h2>Detailed method and source links</h2>
      <details><summary>Coordinate, record, and provenance rules</summary>
        <ul>
          <li>One row represents one canonical encounter within a source year and record series.</li>
          <li>Three multi-sequence archive encounters are aggregated from their component pages; UAV series remain distinct from standard encounters.</li>
          <li>Archive start coordinates are preferred. End coordinates are a fallback only when the start is missing or invalid.</li>
          <li>Unsigned archive longitudes are interpreted as west, subject to the stated plausibility gate; original text and QC remain available.</li>
          <li>Atlist marker response checksums, map update times, record IDs, and source URLs are retained.</li>
        </ul>
      </details>
      <ul>
        <li><a href="{ARCHIVE_INDEX_URL}">CWR Archive Encounters index</a></li>
        <li><a href="https://www.whaleresearch.com/encounters2024">CWR 2024 encounters</a></li>
        <li><a href="https://www.whaleresearch.com/encounters">CWR 2025 encounters</a></li>
        <li><a href="https://www.whaleresearch.com/encounters-map-2026">CWR 2026 encounters</a></li>
      </ul>
    </section>
    <footer>Generated by <code>03_CWR_2017_2026_ALL_ARCHIVES_REPORT_AND_EXPORT.ipynb</code></footer>
    </main>
    </body>
    </html>
    '''

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_html, encoding="utf-8")
    report_sha256 = hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()

    return {
        "report_path": str(REPORT_PATH.resolve()),
        "report_bytes": REPORT_PATH.stat().st_size,
        "report_sha256": report_sha256,
        "map_markers": len(mapped),
        "time_series_months": len(series_months),
        "qc_flagged_records": len(qc_rows),
    }
