# Public observation queries

The base package collects and normalizes killer-whale observations. It does not
require OrcaCast, retained research files, an imputation model, or a Seascape
installation. Python 3.11+ is supported. From a standalone source checkout:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
marine-mammals --workspace-root ./orca-workspace killer-whales observations demo
```

The demo uses two explicitly synthetic local TWM-format records and makes no
network requests. It runs the same collection and normalization API as real queries.
Its outputs are internal-only and are not scientific validation of real providers.

## Query a provider

```bash
marine-mammals --workspace-root ./orca-workspace killer-whales observations sources
marine-mammals --workspace-root ./orca-workspace killer-whales observations query \
  --source inaturalist --start 2025-06-01 --end 2025-06-08 \
  --bbox -126 47 -122 50
```

Bounds are west, south, east, north; dates are inclusive in the configured model
timezone (America/Los_Angeles by default). The default bounds cover the North
Pacific, not the entire world. Supply appropriate bounds for another region.
Antimeridian-crossing boxes must be split into separate queries. Provider date
queries use each provider's date semantics; canonical filtering uses model-local
dates. Query results are reported records, not abundance, survey effort, occupancy,
or verified absence. Location-free records and unsupported uncertainties can be
quarantined; inspect the audit table and normalization manifest.

Select multiple providers by repeating `--source`. The `sources` command lists
provider capabilities and the curated GBIF dataset UUIDs. Choose GBIF datasets:

```bash
marine-mammals --workspace-root ./orca-workspace killer-whales observations query \
  --source gbif --dataset e0da2d53-86f0-440c-a11a-42ffb0b3fd3e \
  --start 2025-06-01 --end 2025-06-08 --bbox -126 47 -122 50
```

Repeat `--dataset` for multiple GBIF datasets. New UUIDs default to internal-use
policy with occurrence-level records; encounter aggregation requires an explicitly
reviewed dataset policy in `--config`. Selecting datasets does not establish their
redistribution rights. The search API path refuses queries exceeding 100,000
records; narrow the interval or bounds. Credentials are read from configured
environment variables; values are not printed. iNaturalist's token is optional.
Acartia can require access credentials depending on provider access policy.

TWM reads local CSV files. Use `--source twm --twm-file /path/to/export.csv`.
If no files exist, collection logs a warning and creates an explicit unavailable
source manifest. Other selected sources continue. Coverage remains unknown, never
verified zero, and prior TWM rows are excluded from the active result while their
history and immutable snapshots remain stored. An existing malformed CSV still
fails validation. Offline replay uses a previously collected TWM snapshot when one
exists; missing offline TWM input is recorded as unavailable.

Acartia's current endpoint and CWR archives are not arbitrary historical date APIs.
Their available observations are filtered to the requested canonical interval, but
that does not make missing historical coverage complete. CWR uses configured
archive years and map IDs. Dataset rights, coverage, and QC remain source-specific.

## Python API

```python
from datetime import date
from marine_mammal_toolkit.cetaceans.killer_whales.query import query_observations

result = query_observations(
    workspace_root="./orca-workspace",
    sources=("inaturalist",),
    start=date(2025, 6, 1),
    end=date(2025, 6, 8),
    bbox=(-126, 47, -122, 50),
)
observations = result.read()
print(result.to_dict())
```

Results include canonical observations, associations, audits, source snapshots, and
`query-result.json`. Each source/dataset/date/bounds configuration uses its own
`data/queries/<query-hash>/` directory. Changed selection cannot reuse another
query's normalization state. Repeating an identical query updates only that query's
workspace; raw snapshots and source history remain available. `--offline` replays
that exact query's local snapshots; it does not silently download missing data.
No observation-query command fits a model, builds counts, prunes results, or
publishes datasets externally.

## Configuration and preflight

A custom YAML may omit unselected providers; omitted providers are disabled.
Source settings still require explicit rights and QC policy. `query --config`
uses that policy while applying the explicit query sources, dates, and bounds.

```bash
marine-mammals --workspace-root ./orca-workspace killer-whales observations preflight
marine-mammals --workspace-root ./orca-workspace killer-whales observations preflight \
  --config /path/to/sightings.yaml --profile imputation-only
```

Preflight checks local configuration and prerequisites without downloads or writes.
It reports absent TWM files as a warning, and missing model-support inputs as
errors for imputation. Provider availability is `not_probed`; this is not a live
API health check. `query --dry-run` prints the selected scope without acquisition.

## Optional modeling and reporting

```bash
python -m pip install '.[report]'
# Seascape must be available from your index or installed from its standalone
# repository before installing the modeling extra:
python -m pip install '.[imputation,report]'
```

Seascape remains a declared optional dependency and supplies the public water
network API. Regional support files and an explicit `SEASCAPE_WORKSPACE` are still
required for marine-distance imputation. This existing binary model is specific
to SRKW/Transient labels with known-Other handling and abstention. Global observation
queries do not imply global ecotype-imputation support. Existing scientific gates
are unchanged. `observations-only` stops after normalization; `observed-only`
also builds counts and requires water-universe inputs.

## Product publication and retention

`product` writes a complete dated generation, including its report, before
advancing `processed/sightings/final/latest.json`. The pointer's manifest, composite,
imputed, and report fields identify immutable paths. Use the public resolver:

```python
from marine_mammal_toolkit.cetaceans.killer_whales.observations import resolve_sightings_product
product = resolve_sightings_product("/absolute/product-root")
# Read product.composite_path and product.imputed_path from this same resolution.
```

The existing flat filenames remain compatibility copies. Their updates roll back
on ordinary write failures, but independent flat-file reads cannot guarantee a
consistent generation during a concurrent write or a process crash. New consumers
must resolve the pointer once. The pointer is committed last and remains consistent
even if copying or report rendering fails. Report rebuilds create a new report
sidecar rather than overwriting a report referenced by an older pointer.

Product and query workspaces reject concurrent writers with a lock file. An
interrupted process can leave `.marine-mammals-write.lock`; inspect its PID/host
and remove it only after confirming that process has stopped. Read-only consumers
do not need this lock. Pruning should run only when no readers depend on generations
that may be removed; pin reproducibility-critical releases.

Pruning is now **off by default**. To explicitly retain three recent generations:

```bash
marine-mammals --workspace-root /absolute/workspace --data-root /absolute/product-root \
  killer-whales observations product --end-date 2025-06-08 \
  --prune --keep-generations 3 --pin-release EXISTING_RELEASE_ID
```

Pins are additional to the recent-generation allowance and are persisted in
`processed/sightings/final/pins.json`. This JSON list may be edited while no writer
is running to unpin a release. Retained generations' model runs and source releases
are protected together. Row-count checks remain an additional deletion guard, not
proof of scientific equivalence and not a gate that rejects the new product.
Raw provider snapshots and source history are never removed by this retention policy.

## Installation validation

CI retains the full scientific tests and adds clean base-wheel installation on
Linux, macOS, and Windows with Python 3.11 and 3.14. The public acceptance script
runs outside the checkout, blocks network access during the demo, and rejects
imports from optional modeling/reporting packages and OrcaCast. A local reproduction:

```bash
python -m pip install build
python -m build --wheel --outdir /temporary/wheels
python scripts/check_public_install.py --wheel-dir /temporary/wheels
```

Dependency installation requires network access; the subsequent demo is offline.
A configured CI matrix is not evidence that every remote runner has passed.
Software licensing is in `LICENSE`; it does not change upstream data rights.
