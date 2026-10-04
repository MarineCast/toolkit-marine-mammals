# Processing and imputation

The killer-whale workflow composes shared observation engines with species-specific
interpretation, features, acceptance policy, and release profiles. The
[architecture and API map](../../../architecture.md) identifies the callable stages.

## Observation processing

Collection retains immutable source snapshots and source-local history.
Normalization applies adapters, canonical identity, deduplication, associations,
quality flags, quarantine, and audit records. Observation queries isolate their
selected sources, datasets, dates, and bounds in separate workspaces.

Preserve source/event identifiers, observation time, coordinate uncertainty,
record grain, and rights when joining or interpreting outputs. Different sources
can have different record grains; rows are not interchangeable animal counts.

## Ecotype label imputation

The implemented binary model infers SRKW/Transient labels for observations. It
preserves known-Other mass, unresolved mass, and abstention. Encounter isolation,
purged evaluation, calibration, support vetoes, and certification gates are part
of the existing workflow. A probability or loaded model does not itself grant
soft-count authority.

Legacy Joblib models are rejected before deserialization and require a refit.
Load only trusted model artifacts. The current model is not a general multiclass
method, an occurrence forecast, or evidence that all populations and regions are
supported.

## Regional support

Marine-distance imputation and intensity use Seascape's public water-network API.
Supply the configured water-network paths and set `SEASCAPE_WORKSPACE` to the
absolute workspace containing Seascape's `config/common.yaml`. Support files and
model-domain polygons must already be provisioned. The bridge reads those products;
it does not build a water network or import a sibling repository's source tree.

Operational model domains describe computation support. They are not claims of
ecological range or critical habitat.

## Release profiles

| Profile | Included stages beyond collection and normalization |
| --- | --- |
| `observations-only` | None; no water universe or imputation model required |
| `imputation-only` | Label imputation |
| `observed-only` | Observed counts; requires water universes |
| `authoritative-counts` | Imputation and daily/weekly counts at configured profile resolutions |
| `production-retrospective` | Imputation, weekly R4 counts, model grids, and intensity; verified cohort required |
| `research-h6` | Imputation, weekly R6 counts, model grids, and intensity |

Profile definitions live in
[the release module](https://github.com/MarineCast/toolkit-marine-mammals/blob/main/src/marine_mammal_toolkit/cetaceans/killer_whales/observations/release.py).
Run `preflight --profile PROFILE` for local prerequisites before materialization.
Release profiles default to internal use; they do not approve external publication.

Counts, grids, and relative reported activity retain their own support and
missingness. Read the [scientific contracts](../../../reference/contracts.md) and
[release guide](products.md) before consuming them.
