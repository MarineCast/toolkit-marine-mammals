> Historical pre-migration reference, retained for scientific context. Old paths, commands,
> schema-version descriptions and execution claims are not current toolkit instructions.
> See [current migration notes](../../migration.md).

# Sightings Pipeline Architecture

The whale package overview, artifact map, imputation semantics, spatial contracts, and
verified commands are documented in
`src/orcacast/domains/whale/README.md` (`../../src/orcacast/domains/whale/README.md`; historical path).
Source-specific contracts, temporal rules, identity resolution, count products, and
operational details are documented in
`docs/data_sources/sightings.md` (`../data_sources/sightings.md`; historical path).

The dependency direction is:

```text
immutable snapshots
  -> source history/current state
  -> canonical observations + associations + identity lineage
  -> optional retrospective soft inference and certified hard imputation
  -> compact imputation fit/calibration/application report
  -> full-area authoritative counts
  -> canonical water cells filtered by provenance-approved operational AOIs
  -> dense ecotype model grids
  -> canonical-water-graph relative reported activity
```

Counts, grids, and intensity are separate durable datasets. Workflows sequence their
public services and pass explicit artifact references; domain code does not invoke CLI
commands or discover inputs from the current working directory.

The current contracts use sightings configuration schema 6, normalized artifact schema
8, imputation schema 8, count/dense-grid Arrow schema 6, and model-grid/intensity
algorithm version 7. `as_of` imputation and counts remain disabled until time-frozen
source state and model fitting are available.

Normalization uses Polars for source adaptation and state reconciliation. Source history
stores payload transitions in append-only compressed Parquet parts, while immutable raw
snapshot manifests preserve every retrieval. A true six-source `--full-refresh` on
2026-08-19 completed in 869.68 seconds / 8.59 GB maximum resident memory. The cost was
dominated by complete provider histories, including per-encounter CWR page fetches and
retries. Forced Polars normalization of that exact manifest completed in 64.01 seconds;
source-state assembly used 44.11 seconds, including 36.04 seconds adapting complete
snapshots. It appended 8,845 payload transitions and produced 170,007 current source
records and 135,652 canonical observations. A provenance-bound model was subsequently
refit on that exact snapshot and applied to all observations. The application retained
probabilities for all 39,335 unknown queries and made zero hard assignments because neither
class passed the purged-blocked certification gate. Counts at H4-H6, the production-scope
weekly H4 model grid (2,490,368 rows), and relative intensity were then rebuilt from explicit
manifests. The preceding incremental collection and normalization took 72.17 and 26.98
seconds, respectively.

Every canonical retrospective imputation application writes a compact HTML report and a
machine-readable JSON companion beside the imputed Parquet artifact. They summarize the
fit sample, selected validation arm, raw versus calibrated Brier/log-loss/ROC-AUC/ECE,
class certification, probability-scored unknowns, and hard imputed points by ecotype.
These reports are manifest outputs and use the same artifact checksum convention as the
prediction table. Soft probability mass is reported separately and is never described as
a hard imputation.

Operational model-domain polygons are built independently from the reviewed SRKW and
Transient AOIs in `config/common.yaml`. The builder validates them against the canonical
H4-H6 full-counting water universes and records exact config, water, geometry, metadata,
and cell-count lineage. These AOIs are operational model extents, not ecological-range or
critical-habitat claims. See
`src/orcacast/domains/whale/SPATIAL_SUPPORT.md` (`../../src/orcacast/domains/whale/SPATIAL_SUPPORT.md`; historical path).
