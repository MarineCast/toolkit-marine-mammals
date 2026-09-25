# Killer-whale research

`sources/cwr` contains source exploration. `imputation` contains retrospective imputation and
experimental calibration, transport, residual and open-set research. These experiments do not
replace the production binary method or confer certification. Occurrence-model research remains
in OrcaCast.

From the toolkit checkout root, install its `research`, `imputation`, and
`report` extras for the imputation notebooks (and make the standalone Seascape
package available), then set explicit storage paths:

```bash
python -m pip install '.[research,imputation,report]'
```

CWR source exploration does not require the imputation extra; its report notebooks
use the `research` and `report` extras.

The imputation notebooks require an existing release containing observations,
imputation diagnostics, and a fitted model; an `observations-only` release or an
isolated public query does not provide those artifacts.

```bash
export MARINE_MAMMALS_WORKSPACE_ROOT=/absolute/existing-data-workspace
export MARINE_MAMMALS_RESEARCH_OUTPUT_ROOT=/absolute/existing-imputation-research-outputs
export MARINE_MAMMALS_CWR_OUTPUT_ROOT=/absolute/existing-cwr-research-directory
export SEASCAPE_WORKSPACE=/absolute/seascape-data-workspace
export MARINE_MAMMALS_WATER_GRID=/absolute/canonical-water-grid.parquet
```

Generated outputs remain in their original locations. Notebook location is not a data-root
discovery mechanism. Start testing notebooks from their `testing` directory to import their
local helper modules. Do not inject OrcaCast or toolkit `src` directories into `sys.path`.

`resolve_release_paths(repo_root=..., release_manifest=...)` accepts an explicit data workspace
and immutable release. The historical `repo_root` parameter now means data workspace; if omitted,
`MARINE_MAMMALS_WORKSPACE_ROOT` is required. Its default release pointer is the retained
OrcaCast layout at `data/processed/domain/whale_layer/sightings/releases/latest.json` under
that workspace. Pass `release_manifest` explicitly when using another release layout; the
notebooks currently call the helper with its default. No parent-repository discovery is performed.

The CWR report helper exposes `build_report(...)` with explicit tables, figures, provenance and
output path; import does not depend on notebook globals or acquire data. Notebook acquisition
cells were not executed during migration. Run research tests with:

```bash
python -m pytest notebooks/cetaceans/killer_whales/imputation/testing --import-mode=prepend -q
```
