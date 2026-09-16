> Historical pre-migration reference, retained for scientific context. Old paths, commands,
> schema-version descriptions and execution claims are not current toolkit instructions.
> See [current migration notes](../../migration.md).

# Killer-Whale Sightings Data Source

## Purpose

The sightings pipeline creates OrcaCast's authoritative killer-whale event and count
datasets. It preserves source reports, resolves them into stable canonical events, and
keeps scientific counts separate from model-domain grids and spatial intensity.

The current artifact boundary is:

```text
collect
  -> assemble complete source history and current state
  -> normalize and resolve canonical events
  -> optionally impute classifications
  -> build authoritative counts
  -> optionally build model-domain grids
  -> optionally calculate spatial intensity
```

Each arrow is an explicit manifest boundary. Services consume `ArtifactRef` inputs;
they do not scan directories for a likely file.

Version numbers describe different contracts and should not be collapsed into one
"pipeline version":

| Contract | Current version |
|---|---|
| Sightings YAML configuration | 6 |
| Normalized observations and state | 7 |
| Imputed observations | 8 |
| Count and dense-grid Arrow schemas | 6 |
| Model-grid and intensity algorithms | 7 |

## Configuration

The authoritative configuration is
`config/data/sightings.yaml` (`../../config/data/sightings.yaml`; historical path). It uses strict schema
version 6. Unknown keys fail validation, lists replace inherited lists, and relative paths
resolve from the repository root.

The configured local sources are:

```yaml
collection:
  sources:
    twm:
      local_path: data/raw/sightings/twm_export
    acartia:
      url: https://acartia.io/api/v1/sightings/current
      local_path: data/raw/sightings/acartia_export
      created_is_event_time: true
    maplify:
      url: https://maplify.com/waseak/php/search-all-sightings.php
    inaturalist:
      url: https://api.inaturalist.org/v1/observations
    cwr:
      archive_years: [2017, 2018, 2019, 2020, 2021, 2022, 2023]
      archive_year_url_template: https://whaleresearch.wixsite.com/{year}encounters
      atlist_api_root: https://api.atlist.com/v1/map
      atlist_maps: {2024: {...}, 2025: {...}, 2026: {...}}
      source_license: UNKNOWN
      source_use_class: INTERNAL_ONLY
    gbif:
      url: https://api.gbif.org/v1/occurrence/search
      taxon_key: 74SZC
      checklist_key: 7ddf754f-d193-4cc9-b351-99906754a03b
      basis_of_record: HUMAN_OBSERVATION
      occurrence_status: PRESENT
      dataset_allowlist: [...]
```

The TWM path is therefore supplied by configuration. Repeatable `--twm-file` options can
override it for one collection. Acartia combines the API response with top-level CSV
files from its configured bulk-export directory. Source-native identity and canonical
event resolution prevent API/export duplication from inflating counts. When one native
ID has different API and bulk payloads in a retrieval, both remain in history, the
current API representation wins current-state selection, and an update is audited.

Credentials are configured by environment-variable name. Values are not written to
manifests or configuration hashes.

## Supported sources

### iNaturalist

The collector requests `Orcinus orca` (taxon ID `41521`) over the complete `full_area`
bbox and paginates at 200 records. Every page must report the same `total_results`, and
the final materialized row count must match it. Incremental refresh begins at the prior
watermark minus a two-day overlap. `--full-refresh` ignores the watermark and requests
history from `min_date`; use it for the initial full-history reconstruction.

### Acartia

The collector snapshots the complete current-endpoint response and copies every
top-level `*.csv` in `data/raw/sightings/acartia_export`. API and CSV rows pass through
the same adapter. Native IDs are selected from `ssemmi_id`, `entry_id`, then `id`.

Acartia is the only source for which `created` is an event timestamp. The source
configuration must explicitly set `created_is_event_time: true`; normalization records
`DATE_BASIS=ACARTIA_CREATED_EVENT`. This is not a generic creation-time fallback.

### TWM

TWM is a local CSV source. Files are read as strings so leading zeros, identifiers, and
date text survive ingestion. Native ID fields are preferred. A row without one receives
a content-based identity derived from stable sighting fields, so renaming a file or
reordering rows does not change identity. The adapter retains `pod`, `likelypod`,
`pod_tag`, J/K/L tags, and positive pod/likely-pod indicator columns as structured
classification evidence.

BC ArcGIS is not a supported source.

### Maplify/WASEAK

The collector requests one configured date/bbox window. The response must contain a
`results` list and a declared `count` equal to the returned row count. Without that
completeness metadata the request fails rather than asserting verified coverage.

### Center for Whale Research (CWR)

CWR is a configured internal-only stream while permission and redistribution terms are
unknown. It contributes to internal canonical observations and counts, but every CWR-only
or CWR-merged observation has `PUBLIC_RELEASE_ELIGIBLE=false`. Public point preparation
filters that field and fails closed when restricted source records lack it.

One complete `FULL_REPLACE` snapshot combines two source systems. The Wix branch reads
the explicitly configured 2017–2023 yearly indexes and every linked encounter page with
eight bounded workers. It stores structured labeled-field extracts, response metadata,
and checksums, but not page HTML, images, or unlabeled narrative text. Standard and UAV
encounters have distinct series. Multi-sequence pages collapse to one source report while
retaining component URLs, checksums, and count. The first valid sequence-start coordinate
is preferred; the final sequence-end coordinate is used only as a documented fallback.
Unsigned archive longitudes are interpreted as west, and candidates outside the configured
CWR plausibility bounds remain source records without usable coordinates.

The Atlist branch refreshes the configured 2024–2026 `fields` and `markers` JSON on every
online run. Encounter markers retain their UUID and raw payload; non-encounter pins are
excluded with raw, accepted, and excluded counts reconciled in snapshot metadata. Stable
native identities are `wix:{year}:{series}:{encounter_number}` and
`atlist:{map_id}:{marker_uuid}`.

Normal online collection reuses the latest immutable Wix extract and refreshes all Atlist
maps. `--full-refresh` refetches both branches; `--offline` reuses the latest complete CWR
snapshot. Missing configured years, failed page requests, malformed JSON, duplicate native
IDs, and reconciliation failures abort collection. Displayed archive years are positive
records only: CWR remains `coverage_status=observed_only` and absence is not zero effort or
verified absence.

CWR observation times are interpreted in `America/Los_Angeles` only when the start is
between 04:00 and 21:00 local and any end is not earlier than the start or more than 16
hours later. Failed gates preserve the raw time text and a QC flag but normalize at date
precision. Ecotype, J/K/L pods and members, and T social groups use the same association
and global 500 m/15 minute identity rules as every other source.

### GBIF, including OBIS-SEAMAP lineage

GBIF is a curated sixth source stream. OrcaCast does not separately ingest SeaMap:
reviewed OBIS-SEAMAP visual datasets arrive through their GBIF dataset keys, which
prevents a second source path from duplicating the same occurrences. The GBIF query is
fixed to Catalogue of Life taxon `74SZC` in checklist
`7ddf754f-d193-4cc9-b351-99906754a03b`, `HUMAN_OBSERVATION`, `PRESENT`, coordinates
present, no flagged geospatial issue, the configured AOI, and `min_date`.

Only committed dataset keys are admitted. The allowlist includes Happywhale North
Pacific, reviewed visual OBIS-SEAMAP datasets, Galiano, and Observation.org. The GBIF
iNaturalist dataset key is explicitly excluded because iNaturalist is collected
directly. Acoustic, machine-observation, specimen, material-sample, and unreviewed
datasets cannot enter through the allowlisted query.

GBIF is collected as `FULL_REPLACE` on every online run. The occurrence endpoint is
paged at 300 rows. Declared totals must remain stable, every `gbifID` must be unique,
and the final row count must reconcile exactly. Queries above the 100,000-result search
limit fail. The immutable snapshot also stores registry metadata and citations,
licenses, OrcaCast dataset policy, request parameters, per-dataset counts, admitted and
quarantined event/occurrence counts, exclusion reasons, file checksums, and raw-schema
fingerprints. Offline mode reuses the selected immutable snapshot.

GBIF occurrence rows are materialized as source events before cross-source identity
resolution. `(datasetKey, eventID)` is used only for datasets whose event semantics were
reviewed. Happywhale may otherwise use exact dataset, timestamp, and coordinates.
Date-only rows never use that heuristic: their identity falls back to `occurrenceID`,
then `gbifID`. A valid grouped event has one day, valid coordinates spanning no more
than 5 km, and no reported coordinate uncertainty above 5 km. Missing uncertainty is
preserved as unknown. Multi-occurrence events retain all occurrence IDs and payloads,
use median coordinates, and carry the maximum reported uncertainty.

Explicit J/K/L members or pods produce SRKW evidence; T-group identifiers and explicit
Bigg's/Transient labels produce Transient evidence. Other individual identifiers are
stored only as dataset-namespaced `INDIVIDUAL` associations. Conflicts are `MIXED`, and
absence of explicit biological evidence remains `UNKNOWN` for imputation. Dataset name,
publisher, geography, and proximity never determine ecotype.

NOAA InPort item 75796 and NOAA's joint-survey feature story describe survey context but
do not expose occurrence records. They remain documentation/provenance references, not
collection inputs.

Reviewed references:

- [OBIS-SEAMAP](https://seamap.env.duke.edu/), scoped to *Orcinus orca*
- [GBIF Orcinus orca occurrence search](https://www.gbif.org/occurrence/search?taxonKey=74SZC)
- [Happywhale North Pacific dataset](https://www.gbif.org/dataset/e0da2d53-86f0-440c-a11a-42ffb0b3fd3e)
- [NOAA InPort item 75796](https://www.fisheries.noaa.gov/inport/item/75796)
- [NOAA joint killer-whale survey feature](https://www.fisheries.noaa.gov/feature-story/first-joint-noaa-killer-whale-survey-examines-endangered-southern-residents-shift)

## Collection and immutable snapshots

Collection writes source material under:

```text
data/raw/whale/sightings/<source>/snapshots/<retrieval-id>/
```

`snapshot.json` records the request, retrieval timestamp, watermark, adapted row count,
raw keys and types, null rates, schema fingerprints, checksums, and explicit temporal
coverage semantics. Exact snapshot files are never edited.

| Snapshot field | Meaning |
|---|---|
| `requested_start`, `requested_through` | The endpoint request interval. |
| `observed_start`, `observed_through` | Bounds found in returned records; descriptive only. |
| `coverage_start`, `coverage_through` | A verified complete interval, or null. |
| `coverage_status` | `request_complete`, `configured_verified`, `observed_only`, or another explicit unverified state. |

Observed bounds never prove source completeness. Maplify, iNaturalist, and GBIF request
intervals are verified only after their response-level completeness checks pass. TWM
Acartia, and CWR remain `observed_only` unless a paired
`verified_coverage_start`/`verified_coverage_through` interval is configured from an
external source contract.

The normalized data-snapshot envelope uses the intersection of verified source
intervals: the latest source start and earliest source end. If any enabled source is
unverified, composite verified coverage is null. Legacy manifests that lack the status
contract load as `legacy_unverified`; asserted dates are not promoted to verified dates.

Remote failure stops collection. `--offline` explicitly selects an existing snapshot and
records that choice. Retries apply only to timeouts, connection errors, HTTP 429, and
HTTP 5xx responses, with exponential backoff, jitter, and `Retry-After` support.
Permanent HTTP 4xx errors fail immediately.

## Complete source state

Normalization does not process only the newest incremental snapshot. It maintains:

| Dataset | Meaning |
|---|---|
| `source_record_history` | first payload plus each later payload transition; immutable snapshot manifests retain every retrieval |
| `source_records` | deterministic current state keyed by `SOURCE_RECORD_ID` |

The history path begins as one Zstandard-compressed Parquet file and is promoted to a
partitioned Parquet dataset on the first appended transition. Unchanged full snapshots do
not add history rows or trigger a multi-gigabyte logical rewrite. Reprocessing an already
applied immutable collect manifest validates and references unchanged source state while
rebuilding the scientific normalization outputs.

Only the scientific outputs and audit are exposed at the dataset root. Incremental and
identity machinery is grouped as internal state:

```text
data/processed/domain/whale_layer/sightings/
├── observations.parquet
├── associations.parquet
├── audit.parquet
├── _state/
│   ├── source_history.parquet
│   ├── source_current.parquet
│   └── identity/
│       ├── assignments.parquet
│       ├── aliases.parquet
│       └── lineage.parquet
└── manifests/
```

Most analysis should read only `observations.parquet` and `associations.parquet`.
`audit.parquet` explains normalization decisions. Files beneath `_state/` are owned by
the pipeline and should not be edited manually.

If an earlier run already wrote the internal tables at the top level, relocate them once:

```bash
orcacast data migrate sightings-state-layout
```

An exact repeated payload is a no-op in current state. A changed payload for the same
native ID creates a `SOURCE_RECORD_UPDATE` audit row. History includes retrieval ID,
retrieval timestamp, payload checksum, and raw-schema fingerprint.

All adapters first emit the same source-neutral fields:

```text
SOURCE_RECORD_ID, SOURCE, SOURCE_NATIVE_ID
OBSERVED_AT_RAW, OBSERVED_DATE_RAW, CREATED_AT_RAW
LATITUDE_RAW, LONGITUDE_RAW
SPECIES_RAW, DESCRIPTION_RAW, POD_ECOTYPE_RAW
SOURCE_DATASET_ID, SOURCE_EVENT_ID
SOURCE_OCCURRENCE_IDS, SOURCE_OCCURRENCE_COUNT
SOURCE_LICENSE, SOURCE_USE_CLASS, COORDINATE_UNCERTAINTY_M
SOURCE_QC_STATUS, SOURCE_QC_DETAIL
SOURCE_PAYLOAD
```

Missing-aware coalescing prevents `NaN`, nulls, and blank strings from hiding valid
fallback values.

## Temporal contract

Canonical event time is intentionally date-based:

| Field | Contract |
|---|---|
| `SIGHTING_DATE` | authoritative model-calendar date |
| `SIGHTING_DATE_UTC` | exactly 12:00:00Z on `SIGHTING_DATE` |
| `SOURCE_EVENT_AT_UTC` | nullable real source event timestamp |
| `SOURCE_CREATED_AT_UTC` | nullable provenance timestamp |
| `DATE_BASIS` | rule that selected the date |
| `SOURCE_TIME_PRECISION` | `DATE` or `TIMESTAMP` |
| `CANONICAL_TIME_SYNTHETIC` | always `true` |

An explicit observation date outranks a timestamp. If they disagree, the explicit date
wins and normalization emits `DATE_TIMESTAMP_CONFLICT`. Timestamp-only dates are derived
in `model_timezone` (`America/Los_Angeles` by default). Ambiguous generic dates are not
guessed. A non-Acartia record with only a creation timestamp is rejected.

Deduplication uses `SOURCE_EVENT_AT_UTC`; counts use `SIGHTING_DATE`. The synthetic noon
timestamp prevents source timezone differences from shifting the counting date.

## Geographic and type normalization

Raw and normalized observations use `full_area`, currently longitude `[-180, -109]` and
latitude `[32, 72]`, covering Alaska through California. Coordinates are swapped only
when the original ordering is invalid and the swapped ordering uniquely lies in the AOI.
Missing dates, invalid coordinates, non-orca species, and records outside the AOI are
retained in the audit with machine-readable rejection reasons.

Every canonical observation has one detailed classification and one exhaustive bucket:

| `ECOTYPE_DETAIL` | `ECOTYPE_BUCKET` |
|---|---|
| `SRKW` | `SRKW` |
| `TRANSIENT` | `TRANSIENT` |
| `NRKW`, `OFFSHORE`, `UNKNOWN`, `MIXED` | `OTHER` |

Evidence precedence is structured ecotype/pod fields, recognized member or social-group
identifiers, non-negated explicit free text, then unknown. A standalone J/K/L is a pod
only in a structured pod field. Free text must say `J pod`, contain a member such as
`J35`, or provide equivalent context. Basic negation suppresses phrases such as
`not SRKW`.

Associations are a separate, provenance-bearing table:

```text
OBSERVATION_ID, SOURCE_RECORD_ID, SOURCE
ASSOCIATION_KIND, ASSOCIATION_VALUE
EVIDENCE_TEXT, EVIDENCE_FIELD, RULE_ID, CONFIDENCE, CONFLICTING
```

T-prefixed identifiers are Transient/Bigg's social groups, not pods. J, K, and L are
SRKW pods. Pod products are nonexclusive because one event may contain multiple pods.

## Deduplication and stable identity

Different native IDs from the same source stay distinct unless a documented
source-specific rule proves equivalence. Cross-source timestamped reports must share the
model date, have compatible detailed ecotypes, be within 15 minutes and 500 metres, and
satisfy complete linkage against every member already in the cluster. Date-only reports
require an exact shared member or transient social-group identifier; broad ecotype text
or pod letters alone cannot merge them.

`SOURCE_REPORT_COUNT` records contributing source-event volume;
`SOURCE_OCCURRENCE_COUNT` separately retains the number of source components represented
by those events, including GBIF occurrence/organism rows and CWR Wix sequence pages.
Canonical event IDs come from
a versioned identity store rather than a hash of current cluster membership:

- a new report joining an event retains the existing ID;
- multiple prior events merging retain the earliest canonical ID and record aliases;
- a prior event splitting creates unique child IDs and `SPLIT` lineage;
- schema-v3 mappings enter v4 as `MIGRATED` lineage;
- duplicate resolved IDs are a hard failure.

Before the clean rebuild, archive schema-v3 state with:

```bash
orcacast data migrate sightings-v3-to-v4
```

Archived files live beneath `data/processed/domain/whale_layer/sightings/legacy/v3/<migration-run>/`
and are lineage only; v4 never resumes them.

## Optional imputation boundary

Imputation remains an explicit, replaceable transform. It consumes normalized
observations and associations and preserves observation identities and order. Its
outputs add observed, imputed, and effective detail/bucket fields plus method,
confidence, temporal direction, and training/evaluation eligibility. Schema v8 also
persists non-null `ENCOUNTER_ID` and `ENCOUNTER_SIZE` from the shared encounter
algorithm. This derived ID is separate from upstream `SOURCE_EVENT_ID`; complete
mapping is required and self-encounter fallback is not allowed.

Each fitted model stores the exact normalized observation checksum, association
checksum, and normalization data-snapshot ID. Application fails if any requested input
differs, so adding a source such as CWR requires refitting from the combined observations and
associations rather than appending rows with an older model.

An imputation run has two possible levels of output:

1. A **soft estimate** supplies `P_SRKW` and `P_TRANSIENT` for a locally supported
   unknown sighting. Eligible estimates may contribute to expected counts.
2. A **hard assignment** populates `ECOTYPE_DETAIL_IMPUTED` and sets
   `IMPUTATION_APPLIED=true`. This requires acceptance by the selective policy,
   prediction stability, and certification of the predicted class by the
   leakage-resistant `purged_blocked` evaluation arm.

The model can therefore run successfully while accepting zero hard assignments. An
abstained row retains `UNKNOWN` as `ECOTYPE_DETAIL_EFFECTIVE`; probabilities are not
silently converted into observed labels. Hard or soft model outputs are never eligible
as independent observed training truth.

Canonical application also writes `imputation_report_retrospective.html` and
`imputation_report_retrospective.json` beside the imputed Parquet file. The deliberately
small report covers model-fit sample sizes, the selected validation/calibration strategy,
raw and calibrated Brier score, log loss, ROC-AUC and ECE, class certification, the number
of probability-scored unknown rows, and hard imputed points split by SRKW and Transient.
The prediction table and both reports are first-class outputs of the imputation manifest.

The configured fit evaluates reconstruction, encounter, blocked, and purged-blocked
splits. `final_calibration_strategy` may use the encounter arm, but
`hard_label_certification_strategy` must be `purged_blocked`. A class is certified only
when the accepted evaluation subset is nonempty and its upper confidence bound on
selective error meets `target_selective_error`.

Retrospective output also records `RETROSPECTIVE_CONTEXT_COMPLETE`,
`IMPUTATION_CONTEXT_STATUS`, and `CONTEXT_COMPLETE_THROUGH_DATE`. An unknown event in
the model's trailing future-context window is `PROVISIONAL_RECENT`; its probabilities
are useful for current maps but are excluded from mature expected counts until a later
run supplies the full configured future window. Observed labels are always treated as
mature count evidence.

Counts can consume either the normalization manifest or an imputation manifest. When
given imputed observations, they retain the compatible integer `SIGHTING_COUNT` based
on the effective hard classification and add probability-aware measures:

| Measure | Meaning |
|---|---|
| `OBSERVED_SIGHTING_COUNT` | records not assigned a hard class by the imputer |
| `HARD_IMPUTED_COUNT` | certified hard assignments |
| `EXPECTED_SIGHTING_COUNT` | summed probability mass for the output ecotype |
| `MATURE_EXPECTED_SIGHTING_COUNT` | observed plus full-context expected mass |
| `PROVISIONAL_EXPECTED_COUNT` | expected mass inside the incomplete recent window |
| `EXPECTED_UNKNOWN_COUNT` | unsupported probability mass retained as unknown |

Model grids carry all measures. Spatial intensity uses
`MATURE_EXPECTED_SIGHTING_COUNT`, so probabilistic local evidence contributes without
silently treating a provisional recent score or an uncertified label as ground truth.

## Redistribution boundary

CWR and GBIF source records retain the applicable source license status and a conservative use class.
CC BY-NC and unknown/unsafe licenses may be used in internal normalized and modeling
artifacts with their attribution metadata, but are `INTERNAL_ONLY`. A canonical event
is public eligible only if every contributing source record is redistribution-safe.
Public point preparation requires `PUBLIC_RELEASE_ELIGIBLE`, treats null as false, and
omits ineligible events. A CWR- or GBIF-bearing artifact without that field fails closed.

## Water universes and routing graphs

Authoritative counts use one common full-area water domain at H3 resolutions 4, 5, and
6. Build it from the validated water polygon product:

```bash
orcacast data build sightings universe \
  --config config/data/sightings.yaml \
  --water-path data/processed/domain/environmental_layer/seascape/spatial_support/water_geometry/TERRITORIAL_WATER_POLYGON.parquet
```

The universe manifest records the source-water checksum, bbox, resolution, cell count,
and output checksum. Counts and model-grid services receive these universes as explicit
`ArtifactRef` inputs.

Normalized observations outside the water universe remain preserved but are excluded
from counts with audit metrics.

The counting universe is not the routing graph. Marine distance and spatial smoothing
load checksum-verified canonical graph products through
`config/data/environment_seascape.yaml`:

- imputation maps observations to the configured R6 or R8 graph, uses canonical
  terminal connectors, and runs bounded shortest-path search;
- known disconnected cells remain unavailable even when an optional straight-line
  fallback is enabled for unknown cells;
- relative reported activity collapses canonical R6 passable edges to R4-R6 adjacency and uses
  bounded graph traversal instead of raw H3 `grid_disk`; and
- probability-surface rendering loads the canonical R8 graph and uses weighted graph
  distance to supported model seeds.

Missing or disconnected routes remain null/unavailable; they are never encoded as zero
distance or infinity.

## Authoritative count products

Daily counts are built first. Weekly counts are derived only by summing daily counts.
The durable products are:

| Dataset | Meaning |
|---|---|
| `ecotype_counts` | mutually exclusive `SRKW`, `TRANSIENT`, `OTHER` counts |
| `ecotype_detail_counts` | QA detail inside those buckets |
| `orca_total_counts` | unique total killer-whale events |
| `pod_counts` | separate nonexclusive J/K/L counts |
| `period_totals` | date-range totals, including explicit zeros |
| `count_exclusions` | observations outside each full-area water universe |

Count rows include `YEAR`, `DAY_OF_YEAR`, `ISO_YEAR`, and `ISO_WEEK`. For weekly rows,
`ISO_YEAR` and `ISO_WEEK` are authoritative; they are never inferred from calendar year
alone. Retrospective runs emit only complete ISO weeks. As-of runs may emit complete
weeks plus one `OPEN` week-to-date period. Partial initial weeks are omitted and reported
in metrics. An empty requested interval succeeds and produces zero-valued period totals.

`SIGHTING_COUNT` counts unique canonical events. `SOURCE_REPORT_COUNT` reports the number
of contributing source reports. The three bucket counts must reconcile exactly to the
separate total product.

## Model grids and intensity

Model grids are downstream of counts and apply ecotype-specific operational model
extents to the canonical full-counting water universes:

| Bucket | Model-grid domain |
|---|---|
| `SRKW` | Canonical water cells whose representative points are covered by the SRKW operational AOI. |
| `TRANSIENT` | Canonical water cells whose representative points are covered by the Transient/Bigg's operational AOI. |
| `OTHER` | Union of both reviewed polygon domains. |

The current polygons are materialized from the reviewed `areas.srkw_range` and
`areas.transient_range` AOIs in `config/common.yaml` by
`config/data/whale/model_domains.yaml`. They are operational analysis extents, not
ecological-range, critical-habitat, presence, or absence evidence. The polygons are not
geometry-clipped to water: the grid service first loads the canonical H3 counting-water
cells and then applies representative-point coverage. The reviewed effective counts are
182/964/5,442 SRKW cells and 421/2,174/12,480 Transient and OTHER cells at H4/H5/H6.

Each polygon requires a sibling `.metadata.json` file containing the ecotype, domain
kind, source authority and repository reference, source release, approval time,
derivation, originating-config checksum, water-membership rule, the exact geometry
checksum, and `review_status=approved_for_model_domain`. Missing, non-polygonal,
unapproved, unsupported-producer, or checksum-mismatched inputs fail closed; the old
implicit bounding-box fallback remains unsupported. See
`src/orcacast/domains/whale/SPATIAL_SUPPORT.md` (`../../src/orcacast/domains/whale/SPATIAL_SUPPORT.md`; historical path).

Each dense partition declares expected cells, periods, rows, period status, and checksum.
Validation requires `rows = cells x periods`; missing a complete cell or period fails.

Spatial intensity is a separate response, not a count, population density, or calibrated
occurrence probability. It uses the configured physical radius in kilometres, traverses
canonical passable-water adjacency, multiplies spatial support by mature ecotype
probability mass, and combines overlapping support with noisy-OR in log-complement
space. A lone fractional classification therefore retains its probability at the source
cell instead of becoming a hard presence. The transform performs no cross-period
smoothing. Manifests record the physical radius, approximate maximum radius, half-weight
distance, maximum supported-cell count, and graph/config lineage.

## Running the pipeline

The examples use canonical `latest.json` pointers; the CLI resolves them to their signed
run manifests.

```bash
# 1. Full source reconstruction (omit --full-refresh for later incremental runs)
orcacast data collect sightings \
  --config config/data/sightings.yaml \
  --full-refresh

# 2. Complete-state assembly, normalization, deduplication, identity resolution
orcacast data process sightings \
  --config config/data/sightings.yaml \
  --input-manifest data/raw/whale/sightings/manifests/latest.json

# 3. Build/verify operational model domains after the counting water universes exist
orcacast data build sightings model-domains \
  --config config/data/sightings.yaml \
  --domain-config config/data/whale/model_domains.yaml

# 4a. Optional periodic model fit (run after normalization when refreshing the model)
MPLCONFIGDIR=/tmp/orcacast-mpl-whale \
orca-impute fit \
  --config config/data/sightings.yaml

# 4b. Apply the latest certified model and write an imputation manifest
orcacast data impute sightings \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/normalize/latest.json \
  --mode retrospective

# 5. Authoritative counts (use the imputation manifest for the imputed flow)
orcacast data build sightings counts \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/impute/retrospective/latest.json \
  --mode retrospective

# 6. Ecotype model-domain dense grids
orcacast data build sightings model-grid \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/counts/latest.json

# 7. Spatial intensity
orcacast data build sightings intensity \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/model_grid/latest.json
```

Use `--dry-run` to inspect configuration, inputs, roots, and planned destinations without
writing. `--force` is required to replace a valid canonical product. `--resume` succeeds
only when the stage signature, code revision, configuration hash, request window,
processing mode, cutoff, input checksums, universe checksums, and validated output
checksums all match.

The CLI exposes the shared processing-mode vocabulary, but `as_of` sightings imputation
and counts currently fail closed until time-frozen source-state reconstruction and a
time-frozen fitted model are implemented. The supported production chain is
`retrospective`.

## Validation and failure handling

Candidate datasets are written under a run-scoped staging directory and validated before
promotion. Invalid candidates move to `data/quarantine/<dataset>/<run-id>/`; a previous
valid canonical artifact remains active. A manifest is written last, so output files by
themselves never indicate a completed stage.

Validation covers Arrow schemas, nullability, identity uniqueness, noon-UTC invariants,
bucket enums, H3 validity/resolution, coordinate bounds, probability pairs and sums,
expected-count conservation, hard-assignment certification and stability, pseudo-label
training exclusion, positive sparse counts, source-report reconciliation, daily/weekly
reconciliation, three-bucket/total reconciliation, dense declared dimensions,
checksums, intensity bounds, and audit coverage.

Inspect registered paths and validate products with:

```bash
orcacast data catalog
orcacast data validate whale.sightings.observations
orcacast data validate whale.sightings.ecotype_counts
orcacast data validate all
```
