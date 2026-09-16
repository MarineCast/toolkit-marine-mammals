> Historical pre-migration reference, retained for scientific context. Old paths, commands,
> schema-version descriptions and execution claims are not current toolkit instructions.
> See [current migration notes](../../migration.md).

# Whale domain

This package owns OrcaCast's durable killer-whale demography and sightings
products. It is a domain-data layer: it preserves source identity and provenance,
normalizes observations, optionally estimates unknown ecotypes, and produces
count and spatial-support products for downstream models. It does not publish an
operational whale-presence forecast by itself.

The canonical processed root is:

```text
data/processed/domain/whale_layer/
├── population/
├── sightings/
└── spatial_support/
```

Do not add compatibility outputs under the removed `data/processed/whale` or
`data/processed/domain/whale` roots.

## Package map

| Path | Responsibility |
|---|---|
| `demography/prepare.py` | Validate the SRKW census workbook and export app-ready annual pod counts. |
| `sightings/collection.py` | Snapshot TWM, Acartia, Maplify, iNaturalist, CWR, and curated GBIF source material. |
| `sightings/adapters.py` | Convert source-native payloads into a common raw record contract. |
| `sightings/normalization.py` | Assemble source history/current state, normalize records, deduplicate events, and preserve stable identities. |
| `sightings/contracts.py` | Arrow schemas and immutable service request types. |
| `sightings/imputation/` | Fit, evaluate, calibrate, certify, and apply selective ecotype inference. |
| `sightings/counts.py` | Build daily and weekly hard and probability-aware count products. |
| `sightings/aggregation.py` | Build polygon-bounded model grids and water-graph relative reported activity. |
| `sightings/spatial.py` | Validate ecotype model-domain polygons and checksum-bound provenance. |
| `sightings/validation.py` | Enforce domain, schema, identity, probability, count, and dense-grid invariants. |
| `sightings/runtime.py` | Stage signatures, resume validation, code revisions, and data-snapshot envelopes. |
| `sightings/service.py` | Small public service façade used by the CLI and workflows. |

The public Python surface is `orcacast.domains.whale.sightings`. Imputation is
loaded lazily by the service façade so ordinary sightings imports do not load the
estimator stack.

## Sightings data flow

```text
immutable source snapshots
  -> source history and deterministic current state
  -> observations, associations, audit, and identity lineage
  -> optional retrospective ecotype inference
  -> H3 R4-R6 daily and weekly counts
  -> ecotype-specific polygon model grids
  -> water-graph relative reported activity
```

Every transition is a manifest boundary. Services consume explicit `ArtifactRef`
objects from the preceding run manifest; they do not select inputs by scanning a
directory.

### Canonical artifacts

| Artifact | Location |
|---|---|
| Normalized observations | `sightings/observations.parquet` |
| Observation associations | `sightings/associations.parquet` |
| Normalization audit | `sightings/audit.parquet` |
| Source state | `sightings/_state/source_{history,current}.parquet` |
| Identity state | `sightings/_state/identity/{assignments,aliases,lineage}.parquet` |
| Retrospective imputation | `sightings/imputed_retrospective.parquet` |
| Imputation report | `sightings/imputation_report_retrospective.{html,json}` |
| Counts | `sightings/counts/mode=retrospective/<product>/` |
| Model grid | `sightings/dense/mode=retrospective/reported_sighting/` |
| Relative reported activity | `sightings/dense/mode=retrospective/relative_reported_activity/` |
| Compatibility alias (one release) | `sightings/dense/mode=retrospective/relative_intensity/` |
| Stage manifests | `sightings/manifests/<stage>/` |

Paths in this table are relative to
`data/processed/domain/whale_layer/`.

## Configuration

`config/data/sightings.yaml` is the strict sightings configuration. It controls:

- source locations, request windows, timezones, and verified-coverage overrides;
- the CWR Wix archive years, Atlist maps, Pacific-time gate, plausibility bounds,
  immutable refresh policy, and internal-use classification;
- the GBIF Orcinus-orca query, reviewed dataset allowlist, iNaturalist exclusion,
  event semantics, uncertainty limits, and redistribution classes;
- the full-area counting bbox and H3 R4-R6 resolutions;
- the canonical seascape water-network configuration;
- SRKW and Transient model-domain polygon paths;
- imputation features, split strategies, calibration, certification, and
  abstention policy; and
- count frequencies and relative-intensity kernel settings.

Unknown configuration keys fail validation. Relative paths resolve from the
repository root.

`config/data/project.yaml` is used by the separate SRKW demography exporter. It
includes `config/data/whale.yaml` and supplies the population source/output paths.

## Provenance and coverage

Source snapshots distinguish four different concepts:

| Field | Meaning |
|---|---|
| `requested_start` / `requested_through` | Interval sent to a source endpoint. |
| `observed_start` / `observed_through` | Minimum and maximum dates found in returned rows. |
| `coverage_start` / `coverage_through` | Interval verified complete by an allowlisted source contract. |
| `coverage_status` | Why coverage is verified or why it remains unverified. |

Observed dates are descriptive and never prove completeness. Legacy manifests
that asserted coverage without the current status contract deserialize as
`legacy_unverified` with null verified bounds. A composite snapshot has verified
coverage only when every enabled source has a verified, overlapping interval.

That snapshot envelope is propagated through normalization, imputation, counts,
model grids, and intensity. Consumers that require current coverage must fail
closed when `coverage_through` is null.

## Ecotype inference: soft estimates and hard assignments

Running the imputer does not mean every unknown sighting receives an ecotype.
There are two distinct outputs:

| Output | Contract |
|---|---|
| Soft probability | `P_SRKW` and `P_TRANSIENT` may be emitted for locally supported unknown observations. These can contribute to explicitly probabilistic expected counts. |
| Hard assignment | `ECOTYPE_DETAIL_IMPUTED` is populated and `IMPUTATION_APPLIED=true` only when the selective policy accepts the prediction, the predicted class is certified by the purged-blocked evaluation arm, and stability checks pass. |

If hard certification fails, the observation remains `UNKNOWN` in
`ECOTYPE_DETAIL_EFFECTIVE`. It may still have probabilities, but it is not used as
hard count evidence and is never eligible as observed training truth.

Important columns:

| Column | Interpretation |
|---|---|
| `ECOTYPE_DETAIL_OBSERVED` | Normalized source classification. |
| `ECOTYPE_DETAIL_IMPUTED` | Nullable accepted hard assignment. |
| `ECOTYPE_DETAIL_EFFECTIVE` | Observed value unless a certified hard assignment was accepted. |
| `IMPUTATION_APPLIED` | Whether a hard assignment changed the effective ecotype. |
| `USE_FOR_HARD_COUNTS` | Whether the row contributes categorical SRKW/Transient count evidence. |
| `USE_FOR_PROBABILISTIC_COUNTS` | Whether probability mass can contribute to expected counts. |
| `CLASS_CERTIFIED_FOR_HARD_LABEL` | Whether the predicted class passed the configured purged-blocked risk guarantee. |
| `ELIGIBLE_FOR_TRAINING` | True only for independent observed training labels, never pseudo-labels. |

The configured fit evaluates reconstruction, encounter, blocked, and
purged-blocked splits. Calibration may use the encounter arm, but hard-label
certification is required to use `purged_blocked`.

## Water and spatial-support contracts

Three spatial concepts are deliberately separate:

1. **Full counting universes** are H3 R4-R6 water-cell sets used to decide which
   normalized observations enter authoritative counts.
2. **Canonical water graphs** come from
   `config/data/environment_seascape.yaml`. Imputation uses the configured graph
   resolution, and intensity collapses canonical R6 passable edges to R4-R6
   adjacency. Known disconnected cells remain unavailable; the code does not
   substitute a land-crossing H3 ring or straight-line distance.
3. **Operational ecotype model domains** are built from the reviewed SRKW and
   Transient AOIs in `config/common.yaml`. Model-grid membership starts with the
   canonical counting-water cells and applies polygon coverage to each H3
   representative point. These AOIs are operational analysis extents, not
   ecological-range or critical-habitat claims. Each polygon requires a sibling
   `.metadata.json` provenance file whose `geometry_sha256` matches the polygon
   and whose `review_status` is `approved_for_model_domain`.

Model-grid construction fails closed if either ecotype polygon or its provenance
is missing, invalid, unapproved, or checksum-mismatched. See
`SPATIAL_SUPPORT.md` (`SPATIAL_SUPPORT.md`; historical path) for the sidecar schema and current
source-status gate.

## Running the sightings pipeline

The CLI accepts the `latest.json` pointer from the preceding stage and resolves
it to the signed run manifest.

```bash
# 1. Collect immutable source snapshots.
orcacast data collect sightings \
  --config config/data/sightings.yaml \
  --full-refresh

# 2. Normalize complete source state and resolve stable observations.
orcacast data process sightings \
  --config config/data/sightings.yaml \
  --input-manifest data/raw/whale/sightings/manifests/latest.json

# 3. Build or verify operational model-domain polygons after water universes exist.
orcacast data build sightings model-domains \
  --config config/data/sightings.yaml \
  --domain-config config/data/whale/model_domains.yaml

# 4. Fit or refresh the reusable retrospective imputer.
MPLCONFIGDIR=/tmp/orcacast-mpl-whale \
orca-impute fit --config config/data/sightings.yaml

# 5. Apply the fitted model. This may emit probabilities while accepting zero
# hard assignments. It also writes a compact HTML/JSON fit, calibration, and
# application report beside the output Parquet file.
orcacast data impute sightings \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/normalize/latest.json \
  --mode retrospective

# 6. Build authoritative hard and expected counts.
orcacast data build sightings counts \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/impute/retrospective/latest.json \
  --mode retrospective

# 7. Build model grids after approved model-domain polygons are installed.
orcacast data build sightings model-grid \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/counts/latest.json

# 8. Build water-graph relative reported activity (and the one-release legacy alias).
orcacast data build sightings intensity \
  --config config/data/sightings.yaml \
  --input-manifest data/processed/domain/whale_layer/sightings/manifests/model_grid/latest.json
```

Before building counts, create or refresh the registered H3 counting universes:

```bash
orcacast data build sightings universe \
  --config config/data/sightings.yaml \
  --water-path data/processed/domain/environmental_layer/seascape/spatial_support/water_geometry/TERRITORIAL_WATER_POLYGON.parquet
```

Use `--dry-run` for read-only resolution, `--resume` only for a matching valid
manifest, and `--force` to replace a canonical product.

`as_of` imputation and counts are not enabled yet: both require time-frozen
source state and a time-frozen fitted model. Use `retrospective` for the current
production pipeline.

## Running the demography exporter

The whale demography exporter is separate from the human-population command:

```bash
python \
  -m orcacast.domains.whale.demography.prepare \
  --config config/data/project.yaml
```

It validates integer census values, unique years, required columns, and reported
all-pods totals. Add `--fail-on-total-mismatch` when a total mismatch should block
the export.

## Validation and tests

```bash
orcacast data catalog
orcacast data validate whale.sightings.observations
orcacast data validate whale.sightings.imputed_retrospective
orcacast data validate whale.sightings.ecotype_counts

MPLCONFIGDIR=/tmp/orcacast-mpl-whale \
python -m pytest -q \
  tests/domains/whale \
  tests/domains/environment/seascape/spatial_support/test_water_network.py
```

Validation is part of artifact promotion. Candidate products are written under a
run-scoped staging directory, checked against Arrow and domain invariants, and
only then atomically replace the canonical artifact. Invalid candidates are
quarantined.

## Release gates

- A source refresh is not verified merely because returned observations have recent
  dates. Every enabled source must provide an allowlisted completeness contract before
  downstream `coverage_through` is non-null.
- Model-grid and intensity regeneration require both approved ecotype polygon artifacts
  and their checksum-matched provenance sidecars.
- A successful imputation run may legitimately accept zero hard assignments. Consumers
  must use `IMPUTATION_APPLIED` and the `USE_FOR_*` flags, not the presence of a model
  version or probability columns, to decide how a row contributes.
- `as_of` imputation and counts remain unavailable until time-frozen source-state and
  model-fitting contracts are implemented.

## More detailed references

- `docs/data_sources/sightings.md` (`../../../../docs/data_sources/sightings.md`; historical path)
  documents source-specific normalization, identity, and counting rules.
- `docs/architecture/sightings_pipeline.md` (`../../../../docs/architecture/sightings_pipeline.md`; historical path)
  summarizes dependency direction.
- `config/data/sightings.yaml` (`../../../../config/data/sightings.yaml`; historical path) is the
  executable configuration contract.
