# OrcaCast extraction — 2026-09-16

## Outcome and ownership

Sightings acquisition, processing, imputation, post-processing and census export now live in
this installable toolkit. Killer-whale configuration, interpretation, imputation features/policy,
release profiles and population presentation compose reusable functions under `tools/`.
See [architecture](architecture.md) and the [file inventory](migration-inventory.json).

The baseline revision was `aba2a9f68452994fe07b72ada6d210f563e48871` in OrcaCast.
Its whale production subtree contained 41 Python files / 17,627 lines, including package markers
(approximately 17,600 implementation lines). The inventory records 90 source, dependency-copy,
test, notebook, configuration and documentation transfers before further function extraction.
Copied shared core support remains in OrcaCast because application consumers still use it.

| Before | After | Reason |
| --- | --- | --- |
| OrcaCast whale producer modules and `orca-impute` | Toolkit tools plus `cetaceans/killer_whales`; `marine-mammals` CLI | Independent source processing and species composition |
| Producer YAMLs under application config | Installed toolkit resources | Wheel execution without either checkout |
| Application-owned request/catalog imports | Toolkit contracts; application catalog adaptation | Explicit producer/consumer ownership |
| Removed seascape source imports | Declared package dependency and public water-network API | No sibling source injection or producer copying |
| Unversioned Joblib models | Versioned toolkit envelope; legacy rejection before deserialization | Refit required; no compatibility unpickling or certification restoration |
| Repository-discovered research inputs | Explicit workspace/output inputs and callable CWR report helper | Research source location does not own generated data |

Occurrence/forecast models, application publishing, forecast evaluation and occurrence-model
research remain in OrcaCast. CWR exploration and imputation research moved beneath
`notebooks/cetaceans/killer_whales`. The historical methods references are retained under
`docs/reference/orcacast-before-migration`; their old commands and execution claims are historical.

## Preserved and intentionally changed contracts

Dataset IDs, `orca:v4:` observation identity, table schemas, source rights, quarantine, coverage
versus zero, population JSON shape, count reconciliation and scientific formulas are retained.
The binary SRKW/Transient model preserves known-Other handling, abstention and purged certification.
The change does not introduce a general multiclass production model.

New producer revisions identify toolkit source/config content rather than the caller's Git cwd.
Configuration includes resolve relative to their declaring file; data/model/output paths require
an explicit workspace. Old manifests remain readable without rewriting their checksums or lineage.
New model envelopes identify the toolkit format; loading always clears soft-count authority.
Only trusted Joblib artifacts may be loaded.

Raw data, generated products, models and generated research exports were left in place. No
production acquisition, refit, artifact rewrite or promotion was run. Source removals are tracked
Git changes and recoverable from the baseline revision; unrelated existing deletions were preserved.

## Validation evidence

- Baseline OrcaCast whale tests: 9 tests collected and 11 collection errors from stale imports of
  the already-removed `orcacast.domains.environment` seascape implementation. This was recorded
  before migration, not reported as a passing baseline.
- Deterministic pre/post extraction comparison passed for canonical rows, audits, clustering,
  observation IDs, identity/alias/lineage tables, associations, five count tables, annual census
  ordering/reconciliation, feature matrices and feature metadata. Five synthetic observations and
  a synthetic water graph were used; only dependency imports were rebound for the old source.
  Reproduce using `scripts/check_extraction_parity.py --legacy-checkout /path/to/OrcaCast`.
- Toolkit plus migrated research: **146 tests passed** (134 toolkit, 12 research). These cover six
  source adapters, pagination/retry containment, identities, quarantine, rights, coverage, temporal
  splits, calibration/certification, unsupported AS_OF, marine disconnection, count/grid/intensity
  semantics, release containment/checksums, population validation and atomic-failure preservation.
- Offline synthetic collection → normalization → counts → temporary release/replay passed. It
  blocks network access. Stable scientific columns match on replay; the existing normalization
  run timestamp `LAST_CORRECTED_AT_UTC` is intentionally excluded from replay equality.
- Deterministic synthetic fitting through encounter and purged-blocked evaluation, conserved
  unknown mass, new-model save/load, and rejection of legacy models before deserialization passed.
- Affected OrcaCast consumers: **54 tests passed**, covering publication contracts, immutable
  sightings consumption, downstream contracts and evaluation. Application CLI help also passed.
- All relocated notebook code cells parse; configuration constructor keywords were checked against
  the installed APIs. No maintained application or toolkit imports reference removed whale modules.

One inherited warning remains: binary imputation uses bitwise inversion of a Python boolean in
an uncertified-mass gate. Python 3.14 deprecates that expression for Python 3.16. It was retained
to avoid silently changing scientific behavior; review it separately before a Python 3.16 upgrade.

## Packaging and execution limits

The wheel includes canonical YAMLs and no data/models. Wheel validation uses a separate virtual
environment outside both repositories, with the toolkit and seascape installed from built wheels.
Existing third-party dependencies are reused without processing their editable-install `.pth`
files; OrcaCast is unavailable and no repository source path is injected. This is an offline
packaging check, not a fresh package-index dependency solve or a multi-Python-version test matrix.
All **92 installed package modules imported**, and **134 toolkit tests passed against the wheel**
from outside either checkout. Installed-wheel CLI checks passed for each command family's help,
packaged configuration loading, a synthetic population workbook export, and an observed-only
release dry-run.

Production-data parity, live provider availability, production model fitting/certification,
regional seascape rebuilding and public promotion were not tested. Research notebooks were
syntax/import checked, not executed against retained production artifacts. Remote publishing and
repository transfer were not performed; repository links were checked against configured remotes,
not remote availability.

Commands used from the owning repositories:

```bash
python -m pytest tests -q
python -m pytest notebooks/cetaceans/killer_whales/imputation/testing --import-mode=prepend -q
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /temporary/wheels .
# OrcaCast, with installed toolkit-human/viewshed dependencies for its consumers:
PYTHONPATH=src python -m pytest tests/test_marine_mammal_publication.py \
  tests/test_sightings_downstream_contracts.py tests/publishing tests/evaluation -q
git diff --check
```
