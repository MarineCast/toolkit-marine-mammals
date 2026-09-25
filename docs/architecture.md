# Ownership and callable stages

| Concern | Reusable implementation | Killer-whale composition |
| --- | --- | --- |
| Acquisition | `tools/observations/collect/sources`, HTTP retry and snapshot pipeline | `configuration.py`: source selection, queries, bounds and rights |
| Processing | `tools/observations/process`: adapters, canonical records, deduplication, identity, quarantine, audit | `observations/interpretation.py`: recognition, ecotypes, evidence and ID prefix |
| Imputation | `tools/observations/impute`: fitting, inference, calibration, splits, encounters, certification, reports and marine routing | `observations/features.py`, `imputation_policy.py`, `imputer.py`, `imputation.py` |
| Post-processing | `tools/observations/post_process`: counts, grids, intensity and spatial operations | Count policy, domain definitions and output mappings |
| Demography | `tools/populations/workbook.py`: headers, integers, annual ordering, reconciliation; shared atomic persistence | `demography/census.py`: SRKW aliases, J/K/L totals and JSON; `populations/prepare.py` remains an import-compatible wrapper |
| Releases | Shared artifact contracts and validators | `pipeline.py`, `observations/release.py`: profiles, orchestration and gates |

Paths are beneath `src/marine_mammal_toolkit`. Contracts live in `tools/schemas`, validation in
`tools/quality`, and the minimal configuration/persistence dependency closure in `tools/_core`.
These are toolkit contracts, not an ecosystem-wide schema package.

Species stage APIs expose `collect`, `process`/`normalize`, `impute`, `counts`, `model_grid`,
`intensity`, and `validate`, using toolkit-owned request/result types. File-oriented training APIs
are in `observations.imputation`: `fit_imputation_model`, `apply_imputation_model`, and
`run_imputation_workflow`. `tools.observations.post_process.pipeline.post_process_observations`
composes counts, grids and intensity from explicit artifact inputs. Population export is
`demography.export_population_numbers`, accepting workspace/source/output and sheet overrides.
See the [demography guide](demography.md) for its CLI and annual-census contract.

The reusable binary engine receives feature building, frame preparation, acceptance policy and
support veto through `ImputationComponents`; fitting workflows accept an imputer factory.
The retained class/diagnostic contract is binary SRKW/Transient with known-Other and unresolved
mass, not arbitrary multiclass classification. Future species must review schema compatibility.
Operational model extents are not ecological-range or critical-habitat claims.

New producer revisions hash installed source/config content independently of caller cwd.
Historical manifests remain readable; provenance changes do not confer release approval.
OrcaCast owns occurrence/forecast modeling, evaluation and publishing, and adapts the toolkit
dataset catalog. No runtime toolkit import may depend on OrcaCast.

The public observation-query facade is `cetaceans/killer_whales/query.py`; it composes the same
collection and normalization stages with isolated query scopes. See [public usage](public-usage.md)
for provider selection, optional modeling dependencies, product resolution, and retention.
