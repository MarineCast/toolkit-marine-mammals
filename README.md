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
│   ├── humpbacks/
│   └── gray_whales/
└── pinnipeds/
    ├── haulouts/
    ├── seals/
    └── sea_lions/
```

Killer-whale species rules, features, acceptance policies, release profiles and population JSON
presentation live under `cetaceans/killer_whales`. The reusable binary engine accepts injected
features/policies; this is not a new general multiclass method.

## Install and run

The package requires Python 3.11 or newer. From this repository:

```bash
python -m pip install -e '.[dev]'
marine-mammals --workspace-root /absolute/data-workspace killer-whales observations run \
  --profile observed-only --end-date 2025-06-08 --dry-run
```

Install the declared `toolkit-seascape` dependency normally (from its own checkout if it is not
available from your package index). No OrcaCast installation or sibling source-path injection is
required. Canonical YAML configurations ship in the wheel. Relative data/model/output paths require
an explicit workspace; includes resolve relative to their declaring file.

Observation commands: `collect`, `process`, `impute fit`, `impute apply`, `post-process counts`,
`post-process model-grid`, `post-process intensity`, `post-process all`, `post-process model-domains`,
`run`, and `validate`. Population export: `killer-whales populations run`.
Use each command's `--help` for manifest inputs and overwrite/resume controls. `run` performs gated
release promotion; it is not a read-only validation command.

Marine-distance imputation and intensity require existing seascape products. Supply
`imputation.inputs.water_network_config` and `water_network_config`, and set `SEASCAPE_WORKSPACE`
to the absolute workspace containing seascape's `config/common.yaml`. The read-only bridge passes
an absolute data base to seascape; it does not copy producers or build water networks.

## Python API

```python
from pathlib import Path
from marine_mammal_toolkit.tools._core.config import workspace
from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import load_sightings_config
from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
    collect, process, impute, counts, model_grid, intensity, validate,
    SightingsCollectionRequest, NormalizationRequest, ImputationRequest,
    CountRequest, ModelGridRequest, IntensityRequest,
)

root = Path('/absolute/data-workspace')
document, settings = load_sightings_config(config_path(), workspace_root=root)
# Supply explicit ArtifactRef inputs and data/artifact/output roots to stage requests.
# Bind the workspace around composed stages that also resolve model/domain paths.
with workspace(root):
    pass  # Invoke only the stages needed for your workflow.
```

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
