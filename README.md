# Marine Mammal Toolkit

Reusable observation and population processing, with an implemented killer-whale workflow
extracted from OrcaCast. Distribution: `marine-mammal-toolkit`; import: `marine_mammal_toolkit`.
Other species remain extension points.

## Package layout

```text
src/marine_mammal_toolkit/
├── tools/
│   ├── observations/   # collect/sources, process, impute, post_process
│   ├── populations/    # Census workbook validation and reconciliation
│   ├── schemas/        # Toolkit-owned requests and artifact contracts
│   ├── quality/        # Validation
│   └── _core/          # Configuration and atomic persistence
├── cetaceans/
│   ├── killer_whales/
│   │   ├── demography/  # SRKW annual census validation and export
│   │   └── populations/ # Existing import path, kept for compatibility
│   ├── humpbacks/
│   └── gray_whales/
└── pinnipeds/
    ├── haulouts/
    ├── seals/
    └── sea_lions/
```

Killer-whale species rules, features, acceptance policies, release profiles and demography JSON
presentation live under `cetaceans/killer_whales`. The reusable binary engine accepts injected
features/policies; this is not a new general multiclass method.

## Install and run

The base package requires Python 3.11+ and supports observation queries without OrcaCast,
Seascape, regional support files, or trained models. From this standalone checkout:

```bash
python -m pip install .
marine-mammals --workspace-root ./orca-workspace killer-whales observations demo
marine-mammals --workspace-root ./orca-workspace killer-whales observations query \
  --source inaturalist --start 2025-06-01 --end 2025-06-08 --bbox -126 47 -122 50
```

The demo is synthetic and offline. Real queries contact the selected providers. Use `sources`
to discover providers and GBIF datasets, `--dataset UUID` to select GBIF datasets, and `preflight`
to check local prerequisites. See **[public usage and installation](docs/public-usage.md)** for
complete CLI/Python examples, provider limitations, result locations, optional dependencies, and
retention. A query writes canonical observations and a provenance manifest beneath its own
`<data-root>/queries/` directory; it does not promote a sightings release.

Canonical YAML configurations ship in the wheel. Paths require an explicit workspace;
configuration includes resolve relative to their declaring file. Existing advanced stage commands
remain available: `collect`, `process`, `impute fit`, `impute apply`, `post-process`, `run`,
`product`, `report`, and `validate`. `run` promotes a local release; it is not read-only.
Install `.[imputation,report]` for the full modeling workflow and `.[dev,imputation,report]`
for the complete test suite. The Seascape optional dependency must be available from your index
or installed from its independent checkout.

### Repository sightings workspace

The checkout-local configuration at `config/killer_whales/sightings.yaml` keeps raw snapshots,
normalized tables, manifests, candidates, models, and immutable releases below
`data/sightings/`. It inherits the packaged scientific configuration.

For existing research workspaces, TWM history and Acartia's supplemental history can optionally
be imported from retained local inputs. The importer expects both source CSV collections; a new
workspace can query other providers without them:

```bash
python scripts/import_legacy_sightings_inputs.py \
  --legacy-data-root /absolute/path/to/OrcaCast/data
```

The importer copies only pipeline-supported TWM and Acartia CSV inputs, preserves Acartia's archive
layout, verifies every copied checksum, and writes
`data/sightings/source_inputs/_import_manifest.json`. It deliberately excludes the unsupported BC
ArcGIS export and redundant iNaturalist CSV exports. These inputs remain ignored, internal-only
data because redistribution rights have not been established; they must not be committed merely
because they are stored inside the checkout. Missing TWM files produce a warning and an explicit
unavailable-source manifest; other selected providers continue. Missing TWM coverage is not treated as zero sightings.

Collect, normalize, and promote the source-observation release without requiring seascape count
universes or an imputation model:

```bash
PYTHONPATH=src python -m marine_mammal_toolkit \
  --workspace-root "$PWD" \
  --data-root data/sightings \
  --artifact-root data/sightings/artifacts \
  --output-root data/sightings/outputs \
  killer-whales observations run \
  --config config/killer_whales/sightings.yaml \
  --profile observations-only \
  --end-date YYYY-MM-DD
```

The promoted pointer is
`data/sightings/processed/sightings/final/releases/latest.json`. The other profiles
continue into imputation and/or counts and retain their existing seascape and certification
prerequisites.

For an application-local product with retained OrcaCast inputs, pass `--product-root` to the same
importer. It requires the historical source CSVs, model domains, and H3 r6 water support, then
copies them under that product's `raw/` tree. A new workspace needs separately provisioned
model and water-support inputs before running `product`; missing TWM files alone do not block
collection. Install `.[imputation,report]` for this command.
The packaged `killer-whales observations product` command then keeps normalization state under
`processed/sightings/normalized/`, model work under `processed/sightings/imputed/`, and publishes
the stable consumer contract under `processed/sightings/final/`. The final directory contains
`composite-sightings.parquet`, `imputed-sightings.parquet`, `imputation-model-manifest.json`, and
`sightings-report.html`, an interactive report with the all-time density of reported sightings and
daily counts over time. Historical products are retained by default. Pass
`--prune --keep-generations 3` to opt in to guarded retention, with `--pin-release ID` to protect additional releases and their model runs.
The row-count guard prevents deletion when counts decrease or increase by more than 10%;
`--max-growth-fraction` adjusts that deletion guard. It does not reject the new product.
Raw provider snapshots and compact stage manifests are not pruned by this policy. Omit `--full-refresh`
for a watermark-based update; ranged APIs use the configured overlap and non-destructive delta upserts.
The imputed table preserves observation identity and abstentions, and remains label imputation—not
occurrence prediction.

The product command owns that internal layout, so it does not require separate artifact or output
roots. Point `--data-root` at the killer-whale product root:

```bash
marine-mammals --workspace-root /absolute/orcacast \
  --data-root data/marine-mammals/killer-whales \
  killer-whales observations product --end-date YYYY-MM-DD
```

The packaged product configuration is the default. To use a custom complete sightings YAML (or a
thin YAML that `extends` another configuration), add the optional flag:

```bash
marine-mammals --workspace-root /absolute/orcacast \
  --data-root data/marine-mammals/killer-whales \
  killer-whales observations product \
  --config config/killer_whales/custom-sightings.yaml \
  --end-date YYYY-MM-DD
```

Relative `--config` paths and configured data paths resolve against `--workspace-root`; relative
`extends` paths resolve against the YAML file that declares them.

It creates and writes this durable structure; temporary `.staging/` and
`_sightings_product_runs/` directories are removed after a successful materialization:

```text
data/marine-mammals/killer-whales/
├── raw/
└── processed/sightings/
    ├── normalized/
    ├── imputed/
    └── final/
```

Rebuild only the HTML from the current stable product, without collection or model fitting:

```bash
marine-mammals --workspace-root /absolute/orcacast \
  --data-root data/marine-mammals/killer-whales \
  killer-whales observations report --force
```

The density map aggregates every valid coordinate to H3 resolution 6 for display. Both charts
describe reported records, not abundance, occupancy, reporting effort, or verified absence; map
tiles are loaded when the report is opened.

Marine-distance imputation and intensity require existing seascape products. Supply
`imputation.inputs.water_network_config` and `water_network_config`, and set `SEASCAPE_WORKSPACE`
to the absolute workspace containing seascape's `config/common.yaml`. The read-only bridge passes
an absolute data base to seascape; it does not copy producers or build water networks.

## Python API

```python
from datetime import date
from marine_mammal_toolkit.cetaceans.killer_whales.query import query_observations

result = query_observations(
    workspace_root="./orca-workspace", sources=("inaturalist",),
    start=date(2025, 6, 1), end=date(2025, 6, 8), bbox=(-126, 47, -122, 50),
)
observations = result.read()
print(result.manifest)
```

Read `latest.json` once or use `resolve_sightings_product()` for a consistent immutable product
generation. Flat product filenames are compatibility copies; see the [publication contract](docs/public-usage.md#product-publication-and-retention).

## Killer-whale demography

The `demography` component validates an annual Southern Resident census workbook and exports
J, K and L pod counts. Supply your own workbook; no census data ships with the package:

```bash
marine-mammals --workspace-root /absolute/workspace killer-whales demography census \
  --workbook /absolute/path/to/census.xlsx --sheet 'Chart Data' \
  --output data/processed/whales/srkw-census.json --dry-run
```

Remove `--dry-run` to write the JSON atomically. Add `--fail-on-total-mismatch` to reject rows
whose reported all-pods count differs from J + K + L. The previous
`killer-whales populations run` command and Python import path remain available to existing
consumers. See the [demography guide](docs/demography.md) for the Python API, input rules and
scientific limits.


Callable clients under `tools.observations.collect.sources` cover TWM, Acartia, Maplify,
iNaturalist, CWR and GBIF. They accept explicit provider settings and injected HTTP callables;
date/geographic filters apply where the source supports them. TWM reads local files; Acartia
snapshots a current endpoint, not a historical date-query API. See [API map](docs/architecture.md).

## Compatibility and validation

Dataset identifiers, observation IDs, schemas, population JSON, rights, missingness and scientific
calculations are retained. New provenance identifies toolkit producer content. Label imputation
is not occurrence forecasting. Legacy Joblib models are rejected before deserialization and must
be refitted; existing files remain untouched. Model loading cannot grant soft-count certification.
Only load trusted models: a version marker does not sandbox Joblib.

```bash
python -m pytest tests -q
python -m pytest notebooks/cetaceans/killer_whales/imputation/testing --import-mode=prepend -q
git diff --check
```

Tests use offline synthetic storage, not production data. See [migration and limits](docs/migration.md)
and [research setup](notebooks/cetaceans/killer_whales/README.md). No raw data, generated research
exports, products, or trained models moved during extraction.
